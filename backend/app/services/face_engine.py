import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class FaceEngine:
    def __init__(
        self,
        detector_path: Optional[str] = None,
        recognizer_path: Optional[str] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(os.path.dirname(base_dir), "models")

        self.detector_path = detector_path or os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        self.recognizer_path = recognizer_path or os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

        if not os.path.exists(self.detector_path):
            raise FileNotFoundError(f"Face detector model not found at: {self.detector_path}")
        if not os.path.exists(self.recognizer_path):
            raise FileNotFoundError(f"Face recognizer model not found at: {self.recognizer_path}")

        # Initialize YuNet Detector (Default input size will be dynamically set on each inference)
        self.detector = cv2.FaceDetectorYN.create(
            model=self.detector_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

        # Initialize SFace Recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.recognizer_path,
            config=""
        )

    def decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode raw image bytes to OpenCV BGR numpy array."""
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image data.")
        return img

    def detect_and_extract(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all faces in image and extracts 128-dimensional embedding for each.
        Returns list of dicts with:
        - bbox: [x, y, w, h]
        - confidence: float
        - landmarks: [[x, y], ...]
        - embedding: list of float (128 elements, L2 normalized)
        """
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

            # Align and crop face for SFace feature extraction
            aligned_face = self.recognizer.alignCrop(img, face)
            feature = self.recognizer.feature(aligned_face)

            # feature is 1x128 array
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
        metric: str = "cosine"
    ) -> Tuple[float, bool]:
        """
        Compares two 128-d embeddings.
        Metric: 'cosine' (Cosine Distance: threshold ~0.363 for SFace)
        or 'norm_l2' (L2 Distance: threshold ~1.128 for SFace).
        Returns: (score, is_match)
        """
        emb1 = np.array([embedding1], dtype=np.float32)
        emb2 = np.array([embedding2], dtype=np.float32)

        if metric == "cosine":
            # SFace match cosine score threshold is 0.363 (lower distance = closer match)
            # cv2.FaceRecognizerSF.match with FR_COSINE returns distance where lower is closer
            score = self.recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)
            # Typically match if score <= 0.363
            is_match = bool(score <= 0.363)
            return float(score), is_match
        else:
            score = self.recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_NORM_L2)
            is_match = bool(score <= 1.128)
            return float(score), is_match

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
