import os
import uuid
from typing import Tuple, Optional
from PIL import Image
import io

class StorageService:
    def __init__(self, upload_dir: str = "uploads"):
        self.driver = os.getenv("STORAGE_DRIVER", "local").lower() # "local" or "s3"
        
        # Local storage configuration
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.upload_dir = os.path.join(base_dir, upload_dir)
        self.raw_dir = os.path.join(self.upload_dir, "raw")
        self.thumb_dir = os.path.join(self.upload_dir, "thumbnails")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.thumb_dir, exist_ok=True)

        # S3 / Object Storage configuration (Works with Biznet GIO NEO, Supabase, Cloudflare R2)
        self.s3_client = None
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "picme-photos")
        self.public_domain = os.getenv("S3_PUBLIC_DOMAIN", "").rstrip("/")
        
        if self.driver == "s3":
            self._init_s3()

    def _init_s3(self):
        try:
            import boto3
            from botocore.config import Config
            
            endpoint_url = os.getenv("S3_ENDPOINT_URL")
            access_key = os.getenv("S3_ACCESS_KEY")
            secret_key = os.getenv("S3_SECRET_KEY")
            region_name = os.getenv("S3_REGION_NAME", "us-east-1")

            self.s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region_name,
                config=Config(signature_version="s3v4")
            )
        except Exception as e:
            print(f"[StorageService] Warning: Failed to init S3 client ({e}). Falling back to local disk.")
            self.driver = "local"

    async def save_photo(
        self, 
        event_id: str, 
        original_filename: str, 
        image_bytes: bytes
    ) -> Tuple[str, str, int, int]:
        """
        Saves original photo and generates an optimized web thumbnail.
        Works seamlessly on both Local disk and S3-compatible cloud storage (Biznet GIO / Supabase / R2).
        Returns: (file_url, thumbnail_url, width, height)
        """
        ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
        unique_name = f"{uuid.uuid4().hex}{ext}"

        # 1. Process thumbnail in-memory with Pillow
        thumb_buffer = io.BytesIO()
        width, height = 0, 0
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            width, height = pil_img.size
            if pil_img.mode in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")
            
            # Thumbnail max 1080px on long edge
            pil_img.thumbnail((1080, 1080), Image.Resampling.LANCZOS)
            pil_img.save(thumb_buffer, "JPEG", quality=80, optimize=True)
            thumb_bytes = thumb_buffer.getvalue()

        # 2. Upload according to active driver
        if self.driver == "s3" and self.s3_client:
            raw_key = f"events/{event_id}/raw/{unique_name}"
            thumb_key = f"events/{event_id}/thumbnails/thumb_{unique_name}"

            # Upload raw
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=raw_key,
                Body=image_bytes,
                ContentType="image/jpeg"
            )

            # Upload thumbnail
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=thumb_key,
                Body=thumb_bytes,
                ContentType="image/jpeg"
            )

            if self.public_domain:
                file_url = f"{self.public_domain}/{raw_key}"
                thumbnail_url = f"{self.public_domain}/{thumb_key}"
            else:
                endpoint = os.getenv("S3_ENDPOINT_URL", "").rstrip("/")
                file_url = f"{endpoint}/{self.bucket_name}/{raw_key}"
                thumbnail_url = f"{endpoint}/{self.bucket_name}/{thumb_key}"

            return file_url, thumbnail_url, width, height

        # Fallback to local disk
        event_raw_dir = os.path.join(self.raw_dir, event_id)
        event_thumb_dir = os.path.join(self.thumb_dir, event_id)
        os.makedirs(event_raw_dir, exist_ok=True)
        os.makedirs(event_thumb_dir, exist_ok=True)

        raw_path = os.path.join(event_raw_dir, unique_name)
        thumb_path = os.path.join(event_thumb_dir, f"thumb_{unique_name}")

        with open(raw_path, "wb") as f:
            f.write(image_bytes)

        with open(thumb_path, "wb") as f:
            f.write(thumb_bytes)

        file_url = f"/uploads/raw/{event_id}/{unique_name}"
        thumbnail_url = f"/uploads/thumbnails/{event_id}/thumb_{unique_name}"

        return file_url, thumbnail_url, width, height
