import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from backend.app.services.face_engine import FaceEngine
from backend.app.services.storage import StorageService
from backend.app.services.database import DatabaseService

app = FastAPI(
    title="PixMe API",
    description="AI-Powered Event Photo Face-Matching Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
face_engine = FaceEngine()
storage_service = StorageService()
db_service = DatabaseService()

# Mount uploaded media directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
uploads_path = os.path.join(base_dir, "uploads")
os.makedirs(uploads_path, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_path), name="uploads")

# Mount web frontend static assets if exists
frontend_path = os.path.join(base_dir, "frontend")
os.makedirs(frontend_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


class CreateEventRequest(BaseModel):
    title: str
    location: str
    event_date: str
    category: Optional[str] = "music"
    description: Optional[str] = ""


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "PixMe AI Engine"}


@app.get("/api/events")
def list_events(category: Optional[str] = None):
    events = db_service.list_events()
    if category and category != "all":
        events = [e for e in events if e.get("category") == category]
    return {"events": events}


@app.post("/api/events")
def create_event(req: CreateEventRequest):
    event = db_service.create_event(
        title=req.title,
        location=req.location,
        event_date=req.event_date,
        category=req.category or "music",
        description=req.description or ""
    )
    return {"event": event}


@app.get("/api/events/{event_id}")
def get_event(event_id: str):
    event = db_service.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"event": event}


@app.get("/api/events/{event_id}/photos")
def list_photos(event_id: str):
    photos = db_service.list_photos_by_event(event_id)
    return {"photos": photos, "total": len(photos)}


@app.post("/api/events/{event_id}/photos")
async def upload_photos(event_id: str, files: List[UploadFile] = File(...)):
    """
    Ingests event photos from photographers:
    1. Saves original + generates optimized thumbnail
    2. Detects all faces via YuNet (CPU)
    3. Extracts 128-d embeddings via SFace
    4. Computes pHash for duplicate prevention
    """
    event = db_service.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    processed = []
    for file in files:
        contents = await file.read()
        try:
            # Decode for AI processing
            img = face_engine.decode_image(contents)
            phash = face_engine.compute_phash(img)

            # Detect faces & extract embeddings
            faces = face_engine.detect_and_extract(img)

            # Save media
            file_url, thumb_url, w, h = await storage_service.save_photo(
                event_id=event_id,
                original_filename=file.filename or "photo.jpg",
                image_bytes=contents
            )

            # Save to Database
            record = db_service.save_photo(
                event_id=event_id,
                filename=file.filename or "photo.jpg",
                file_url=file_url,
                thumbnail_url=thumb_url,
                phash=phash,
                width=w,
                height=h,
                size_bytes=len(contents),
                faces=faces
            )
            processed.append(record)
        except Exception as e:
            processed.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"uploaded": processed, "total": len(processed)}


@app.post("/api/events/{event_id}/search")
async def search_faces(
    event_id: str,
    selfie: UploadFile = File(...),
    distance_threshold: float = Form(0.38)
):
    """
    User Face-Search:
    1. Uploads selfie photo
    2. Detects user face & extracts query embedding
    3. Runs cosine distance vector search across event photos
    4. Returns matched photos ranked by similarity
    """
    event = db_service.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    contents = await selfie.read()
    img = face_engine.decode_image(contents)

    faces = face_engine.detect_and_extract(img)
    if not faces:
        return {
            "matches": [],
            "message": "No face detected in selfie. Please ensure good lighting and clear face visibility."
        }

    # Use the largest face if multiple are present in selfie
    best_face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
    query_embedding = best_face["embedding"]

    matches = db_service.search_faces(
        event_id=event_id,
        query_embedding=query_embedding,
        distance_threshold=distance_threshold,
        limit=50
    )

    return {
        "matches": matches,
        "total_matches": len(matches),
        "selfie_bbox": best_face["bbox"]
    }


class PurchaseRequest(BaseModel):
    user_id: str
    tier: str = "high_res"  # "high_res" (Rp 35.000) or "compressed" (Rp 25.000)


@app.post("/api/photos/{photo_id}/purchase")
async def purchase_photo(photo_id: str, req: PurchaseRequest):
    """
    Handles tiered photo purchase:
    - High-Res (Rp 35.000): Preserves high-res original 24-50MP master file in storage.
      User can download either high-res master or compressed version.
    - Compressed (Rp 25.000): Automatically deletes raw high-res original
      from storage to optimize cloud storage COGS, preserving only compressed web version.
    """
    photo = db_service.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    purchase = db_service.record_purchase(
        user_id=req.user_id,
        photo_id=photo_id,
        tier=req.tier
    )

    if req.tier == "compressed":
        # Delete high-resolution original to save storage space!
        storage_service.delete_raw_photo(photo["file_url"])
        db_service.mark_photo_raw_deleted(photo_id)
        download_url = photo["thumbnail_url"]
        high_res_available = False
    else:
        # High-res master preserved
        download_url = photo["file_url"]
        high_res_available = True

    return {
        "success": True,
        "purchase": purchase,
        "tier": req.tier,
        "high_res_available": high_res_available,
        "download_url": download_url,
        "compressed_download_url": photo["thumbnail_url"],
        "message": "Pembelian berhasil!"
    }


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_html = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_html):
        with open(index_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PixMe Backend is running. Please add frontend/index.html</h1>"


@app.get("/admin", response_class=HTMLResponse)
@app.get("/admin/{path:path}", response_class=HTMLResponse)
def admin_page(path: Optional[str] = None):
    admin_html = os.path.join(frontend_path, "admin.html")
    if os.path.exists(admin_html):
        with open(admin_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PixMe Master Admin Dashboard not found.</h1>"
