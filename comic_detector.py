"""
comic_detector.py

Wraps ogkalu/comic-text-and-bubble-detector (RT-DETR-v2, hosted on Hugging
Face), a model fine-tuned on ~11k manga/webtoon/manhua/western-comic images
specifically for text and bubble detection. This replaces page-wide
classical-CV bubble detection with a detector actually trained for this
problem, and gives us the "text_free" (SFX/caption, outside any bubble)
label directly instead of inferring it from shape heuristics.

Classes (from the model card):
  0: bubble       -- the bubble outline itself
  1: text_bubble  -- text lines *inside* a bubble
  2: text_free    -- text lines *outside* any bubble (SFX, captions, signage)

Only `bubble` (for shape/legend + OCR region) and `text_bubble` (to tighten
the OCR crop) feed into the pipeline's output; `text_free` is detected but
excluded by default -- this is how "no SFX" is enforced structurally rather
than heuristically.

NOTE: this model was trained at 640px input resolution, with tall webtoon
pages split vertically before detection. We replicate that here by slicing
the page into overlapping chunks, running detection per chunk, and merging
boxes back into full-page coordinates with NMS-based deduplication.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image

MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
LABELS = {0: "bubble", 1: "text_bubble", 2: "text_free"}


@dataclass
class Detection:
    label: str
    score: float
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1, full-page coords


class ComicDetector:
    def __init__(self, score_threshold: float = 0.3):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as e:
            raise ImportError("Run: pip install transformers torch") from e
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.score_threshold = score_threshold
        # Downloads from Hugging Face on first run -- needs internet access once.
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).to(self.device).eval()

    def _detect_chunk(self, pil_chunk: Image.Image) -> List[Detection]:
        torch = self._torch
        inputs = self.processor(images=pil_chunk, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil_chunk.size[::-1]])
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=target_sizes
        )[0]
        dets = []
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            dets.append(Detection(
                label=LABELS.get(int(label_id), f"unknown_{int(label_id)}"),
                score=float(score),
                bbox=(x0, y0, x1, y1),
            ))
        return dets

    def detect(self, image_bgr: np.ndarray, chunk_size: int = 1024, overlap: int = 150) -> List[Detection]:
        """Slice a (possibly very tall webtoon) page into overlapping chunks,
        run detection per chunk, and merge results into full-page coordinates."""
        import cv2
        H, W = image_bgr.shape[:2]
        all_dets: List[Detection] = []

        y = 0
        while y < H:
            y_end = min(y + chunk_size, H)
            chunk_bgr = image_bgr[y:y_end]
            chunk_rgb = cv2.cvtColor(chunk_bgr, cv2.COLOR_BGR2RGB)
            pil_chunk = Image.fromarray(chunk_rgb)

            for det in self._detect_chunk(pil_chunk):
                x0, y0, x1, y1 = det.bbox
                all_dets.append(Detection(
                    label=det.label, score=det.score,
                    bbox=(x0, y0 + y, x1, y1 + y),
                ))

            if y_end == H:
                break
            y += chunk_size - overlap

        return self._dedup(all_dets)

    @staticmethod
    def _iou(a, b) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _dedup(cls, dets: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
        """Greedy per-label NMS, since the same box can appear twice across
        overlapping chunk boundaries."""
        by_label = {}
        for d in dets:
            by_label.setdefault(d.label, []).append(d)

        kept: List[Detection] = []
        for label, group in by_label.items():
            group.sort(key=lambda d: d.score, reverse=True)
            chosen: List[Detection] = []
            for d in group:
                if any(cls._iou(d.bbox, c.bbox) > iou_thresh for c in chosen):
                    continue
                chosen.append(d)
            kept.extend(chosen)
        return kept
