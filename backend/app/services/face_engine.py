import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class FaceEngine:
    """
    AI Face Recognition Engine for PicMe.
    Primary Engine: InsightFace SOTA (SCRFD Face Detector + ArcFace Recognizer, 512-d embeddings).
    Fallback Engine: OpenCV YuNet + SFace (128-d embeddings).
    """
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640)
    ):
        self.use_insightface = False
        self.insightface_app = None
        self.det_size = det_size

        # 1. Attempt to initialize InsightFace (SCRFD + ArcFace)
        try:
            from insightface.app import FaceAnalysis
            print(f"[FaceEngine] Initializing InsightFace SOTA engine ({model_name})...")
            self.insightface_app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            self.insightface_app.prepare(ctx_id=0, det_size=self.det_size)
            self.use_insightface = True
            print("[FaceEngine] InsightFace (SCRFD + ArcFace 512-d) loaded successfully!")
        except Exception as e:
            print(f"[FaceEngine] InsightFace initialization skipped/failed: {e}")
            print("[FaceEngine] Falling back to OpenCV YuNet + SFace...")

        # 2. Fallback to OpenCV YuNet + SFace if InsightFace not available
        if not self.use_insightface:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(os.path.dirname(base_dir), "models")

            detector_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
            recognizer_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

            if os.path.exists(detector_path) and os.path.exists(recognizer_path):
                self.detector = cv2.FaceDetectorYN.create(
                    model=detector_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000
                )
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=recognizer_path,
                    config=""
                )
            else:
                print("[FaceEngine] Warning: Fallback model files not found.")

    def decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode raw image bytes to OpenCV BGR numpy array."""
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image data.")
        return img

    def detect_and_extract(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all faces in image and extracts normalized embedding for each.
        - InsightFace: SCRFD detector + ArcFace 512-dimensional vector.
        - YuNet: YuNet detector + SFace 128-dimensional vector.
        Returns list of dicts with:
        - bbox: [x, y, w, h]
        - confidence: float
        - landmarks: [[x, y], ...]
        - embedding: list of float (L2 normalized)
        """
        # Primary: InsightFace (SCRFD + ArcFace)
        if self.use_insightface and self.insightface_app is not None:
            faces = self.insightface_app.get(img)
            if not faces:
                return []

            results = []
            for face in faces:
                # Bounding box in [x1, y1, x2, y2] -> convert to [x, y, w, h]
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                bbox = [x1, y1, w, h]

                conf = float(face.det_score) if hasattr(face, 'det_score') else 1.0
                landmarks = face.kps.tolist() if hasattr(face, 'kps') and face.kps is not None else []

                # ArcFace 512-d feature vector
                embedding = face.normed_embedding.tolist()

                results.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "landmarks": landmarks,
                    "embedding": embedding
                })
            return results

        # Fallback: OpenCV YuNet + SFace
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            bbox = [int(val) for val in face[:4]]
            conf = float(face[-1])
            landmarks = [[float(face[i]), float(face[i+1])] for i in range(4, 14, 2)]

            aligned_face = self.recognizer.alignCrop(img, face)
            feature = self.recognizer.feature(aligned_face)
            embedding = feature[0].tolist()

            results.append({
                "bbox": bbox,
                "confidence": conf,
                "landmarks": landmarks,
                "embedding": embedding
            })

        return results

    def compare_faces(
        self, 
        embedding1: List[float], 
        embedding2: List[float], 
        threshold: Optional[float] = None
    ) -> Tuple[float, bool]:
        """
        Compares two face embeddings using Cosine Similarity.
        For ArcFace (512-d): threshold ~0.50 (cosine similarity >= 0.50 is match).
        Returns: (similarity_score, is_match)
        """
        emb1 = np.array(embedding1, dtype=np.float32)
        emb2 = np.array(embedding2, dtype=np.float32)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 > 0:
            emb1 /= norm1
        if norm2 > 0:
            emb2 /= norm2

        cosine_sim = float(np.dot(emb1, emb2))

        # Default threshold: 0.50 for ArcFace (512-d) or 0.637 for SFace (128-d)
        if threshold is None:
            threshold = 0.50 if len(embedding1) > 256 else 0.637

        is_match = cosine_sim >= threshold
        return cosine_sim, is_match

    @staticmethod
    def compute_phash(img: np.ndarray, hash_size: int = 8) -> str:
        """
        Computes 64-bit perceptual hash (pHash) to detect identical/duplicate photos.
        """
        resized = cv2.resize(img, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:hash_size, :hash_size]
        avg = (dct_low.sum() - dct_low[0, 0]) / (hash_size * hash_size - 1)
        phash_bits = dct_low > avg
        phash_hex = "".join(f"{b:02x}" for b in np.packbits(phash_bits))
        return phash_hex
