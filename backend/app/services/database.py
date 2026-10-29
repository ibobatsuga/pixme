import os
import sqlite3
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional

class DatabaseService:
    def __init__(self, db_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = db_path or os.path.join(base_dir, "eventphoto.db")
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                slug TEXT UNIQUE NOT NULL,
                category TEXT DEFAULT 'music',
                description TEXT,
                location TEXT NOT NULL,
                event_date TEXT NOT NULL,
                cover_image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_url TEXT NOT NULL,
                thumbnail_url TEXT NOT NULL,
                phash TEXT,
                width INTEGER,
                height INTEGER,
                size_bytes INTEGER,
                faces_detected INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id TEXT PRIMARY KEY,
                photo_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
                FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS purchases (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                photo_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                price_idr INTEGER NOT NULL,
                photographer_share_idr INTEGER NOT NULL,
                eo_share_idr INTEGER NOT NULL,
                platform_share_idr INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (photo_id) REFERENCES photos (id)
            )
            """)
            # Migration check: add raw_deleted to photos
            try:
                cursor.execute("ALTER TABLE photos ADD COLUMN raw_deleted INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    def create_event(self, title: str, location: str, event_date: str, description: str = "", category: str = "music") -> Dict[str, Any]:
        event_id = str(uuid.uuid4())
        slug = title.lower().replace(" ", "-") + "-" + event_id[:8]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Ensure column exists for existing database files
            try:
                cursor.execute("ALTER TABLE events ADD COLUMN category TEXT DEFAULT 'music'")
                conn.commit()
            except sqlite3.OperationalError:
                pass

            cursor.execute(
                "INSERT INTO events (id, title, slug, category, location, event_date, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, title, slug, category, location, event_date, description)
            )
            conn.commit()
        return self.get_event(event_id)

    def list_events(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE id = ? OR slug = ?", (event_id, event_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_photo(
        self,
        event_id: str,
        filename: str,
        file_url: str,
        thumbnail_url: str,
        phash: str,
        width: int,
        height: int,
        size_bytes: int,
        faces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        photo_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO photos 
                (id, event_id, filename, file_url, thumbnail_url, phash, width, height, size_bytes, faces_detected) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (photo_id, event_id, filename, file_url, thumbnail_url, phash, width, height, size_bytes, len(faces))
            )
            for face in faces:
                face_id = str(uuid.uuid4())
                cursor.execute(
                    """INSERT INTO face_embeddings 
                    (id, photo_id, event_id, embedding_json, bbox_json, confidence) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        face_id, 
                        photo_id, 
                        event_id, 
                        json.dumps(face["embedding"]), 
                        json.dumps(face["bbox"]), 
                        face["confidence"]
                    )
                )
            conn.commit()
        return {
            "id": photo_id,
            "event_id": event_id,
            "filename": filename,
            "file_url": file_url,
            "thumbnail_url": thumbnail_url,
            "faces_detected": len(faces)
        }

    def list_photos_by_event(self, event_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM photos WHERE event_id = ? ORDER BY created_at DESC", (event_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_faces(
        self,
        event_id: str,
        query_embedding: List[float],
        distance_threshold: float = 0.363,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine distance between query_embedding and all face embeddings for the given event.
        Cosine distance for normalized vectors: 1 - (u . v)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT fe.photo_id, fe.embedding_json, fe.bbox_json, p.file_url, p.thumbnail_url, p.filename
                FROM face_embeddings fe
                JOIN photos p ON fe.photo_id = p.id
                WHERE fe.event_id = ?
                """,
                (event_id,)
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        matches = {}
        for row in rows:
            emb = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb /= emb_norm

            # Cosine distance: 1 - cosine_similarity
            cos_sim = float(np.dot(q_vec, emb))
            cos_dist = 1.0 - cos_sim

            if cos_dist <= distance_threshold:
                photo_id = row["photo_id"]
                # If a photo has multiple matches (e.g. user appears multiple times), keep best score
                if photo_id not in matches or cos_dist < matches[photo_id]["distance"]:
                    matches[photo_id] = {
                        "photo_id": photo_id,
                        "file_url": row["file_url"],
                        "thumbnail_url": row["thumbnail_url"],
                        "filename": row["filename"],
                        "distance": round(cos_dist, 4),
                        "similarity": round(max(0.0, cos_sim), 4),
                        "bbox": json.loads(row["bbox_json"])
                    }

        sorted_matches = sorted(matches.values(), key=lambda x: x["distance"])
        return sorted_matches[:limit]

    def get_photo(self, photo_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def mark_photo_raw_deleted(self, photo_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE photos SET raw_deleted = 1 WHERE id = ?", (photo_id,))
            conn.commit()

    def record_purchase(
        self,
        user_id: str,
        photo_id: str,
        tier: str = "high_res"
    ) -> Dict[str, Any]:
        """
        Records a purchase for high_res (Rp 35.000) or compressed (Rp 20.000).
        Revenue share: Photographer 50%, Venue/EO 10%, Platform PicMe 40%.
        """
        photo = self.get_photo(photo_id)
        if not photo:
            raise ValueError(f"Photo {photo_id} not found")

        price = 35000 if tier == "high_res" else 20000
        photographer_share = int(price * 0.50)
        eo_share = int(price * 0.10)
        platform_share = price - photographer_share - eo_share

        purchase_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO purchases (id, user_id, photo_id, event_id, tier, price_idr, photographer_share_idr, eo_share_idr, platform_share_idr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (purchase_id, user_id, photo_id, photo["event_id"], tier, price, photographer_share, eo_share, platform_share)
            )
            conn.commit()

        return {
            "purchase_id": purchase_id,
            "photo_id": photo_id,
            "event_id": photo["event_id"],
            "tier": tier,
            "price_idr": price,
            "photographer_share_idr": photographer_share,
            "eo_share_idr": eo_share,
            "platform_share_idr": platform_share
        }
