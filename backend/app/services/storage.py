import os
import uuid
from typing import Tuple
from PIL import Image
import io

class StorageService:
    def __init__(self, upload_dir: str = "uploads"):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.upload_dir = os.path.join(base_dir, upload_dir)
        self.raw_dir = os.path.join(self.upload_dir, "raw")
        self.thumb_dir = os.path.join(self.upload_dir, "thumbnails")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.thumb_dir, exist_ok=True)

    async def save_photo(
        self, 
        event_id: str, 
        original_filename: str, 
        image_bytes: bytes
    ) -> Tuple[str, str, int, int]:
        """
        Saves original photo and generates an optimized web thumbnail.
        Returns: (file_url, thumbnail_url, width, height)
        """
        ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
        unique_name = f"{uuid.uuid4().hex}{ext}"

        event_raw_dir = os.path.join(self.raw_dir, event_id)
        event_thumb_dir = os.path.join(self.thumb_dir, event_id)
        os.makedirs(event_raw_dir, exist_ok=True)
        os.makedirs(event_thumb_dir, exist_ok=True)

        raw_path = os.path.join(event_raw_dir, unique_name)
        thumb_path = os.path.join(event_thumb_dir, f"thumb_{unique_name}")

        # Save raw original
        with open(raw_path, "wb") as f:
            f.write(image_bytes)

        # Generate lightweight thumbnail with Pillow
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            width, height = pil_img.size
            # Convert to RGB if RGBA
            if pil_img.mode in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")

            # Thumbnail max 1080px on long edge
            pil_img.thumbnail((1080, 1080), Image.Resampling.LANCZOS)
            pil_img.save(thumb_path, "JPEG", quality=80, optimize=True)

        file_url = f"/uploads/raw/{event_id}/{unique_name}"
        thumbnail_url = f"/uploads/thumbnails/{event_id}/thumb_{unique_name}"

        return file_url, thumbnail_url, width, height
