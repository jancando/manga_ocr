"""
ocr.py

OCR backends for reading text out of a detected bubble. Both backends share
the same masked-crop strategy: everything *outside* the bubble's contour is
painted white before OCR runs, so:
  1. Overlapping panel art / neighboring bubbles don't leak into the input.
  2. SFX is excluded by construction -- SFX text isn't inside any detected
     bubble contour, so it's never shown to the OCR model at all.

Backends:
  - EasyOCRBackend  : general-purpose, supports ~80 languages -- use this
                       for anything that isn't pure Japanese, or for mixed-
                       language material.
  - MangaOCRBackend : Japanese-only, but a transformer trained specifically
                       on manga typography (vertical text, furigana,
                       stylized fonts) -- noticeably more accurate than
                       EasyOCR on Japanese specifically. Use when you know
                       a page/series is Japanese.

pipeline.py picks a backend based on the --backend / --langs CLI flags.
"""

from typing import List
import numpy as np
import cv2
from PIL import Image

from bubble_detector import Bubble


def _masked_crop(image_bgr: np.ndarray, bubble: Bubble, padding: int = 6) -> np.ndarray:
    x, y, w, h = bubble.bbox
    H, W = image_bgr.shape[:2]
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(W, x + w + padding), min(H, y + h + padding)

    crop = image_bgr[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    shifted_contour = bubble.contour - [x0, y0]
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    white_bg = np.full_like(crop, 255)
    return np.where(mask[..., None] == 255, crop, white_bg)


class PaddleOCRBackend:
    """
    PP-OCRv6 text recognition models based on PPLCNetV4 + LightSVTR + CTC/NRTR multi-head decoder, 
    single model supports 50 languages (tiny: 49). Medium achieves +5.1% over PP-OCRv5_server.
    https://github.com/PaddlePaddle/PaddleOCR
    """

    def __init__(self):
        try:
            from paddleocr import PaddleOCR
            import logging
            logging.getLogger("paddlex").setLevel(logging.ERROR)
        except ImportError as e:
            raise ImportError("Run: pip install paddleocr") from e
        
        import torch

        self._model = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu:0" if torch.cuda.is_available() else "cpu",
            engine="transformers",
        )

    def read_bubble(self, image_bgr: np.ndarray, bubble: Bubble, padding: int = 6) -> str:
        masked = _masked_crop(image_bgr, bubble, padding)
        results = self._model.predict(masked)
        texts = []
        for result in results:
            data = result.json
            if isinstance(data, str):
                import json
                data = json.loads(data)

            # PaddleOCR 3.x wraps recognition output in "res".
            res = data.get("res", data)
            rec_texts = res.get("rec_texts", [])

            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]

            texts.extend(text for text in rec_texts if text.strip())

        return " ".join(texts).strip()

    def read_all(self, image_bgr: np.ndarray, bubbles: List[Bubble]) -> List[str]:
        return [self.read_bubble(image_bgr, b) for b in bubbles]


class EasyOCRBackend:
    """
    General-purpose multi-language OCR. Pass the language(s) you expect,
    e.g. langs=['en'], langs=['fr'], langs=['ko'], langs=['ch_sim', 'en'].

    IMPORTANT -- EasyOCR groups languages into compatible sets: most Latin-
    script languages can be combined freely (e.g. ['en','fr','de']), but
    Japanese/Korean/Chinese each only combine with English, not with each
    other or with most other scripts. If your source mixes e.g. Japanese
    and Korean on the same pages, run the pipeline twice with different
    --langs and merge results, or pick the dominant language.
    Full compatibility list: https://www.jaided.ai/easyocr/
    """

    def __init__(self, langs: List[str] = None, gpu: bool = False):
        try:
            import easyocr
        except ImportError as e:
            raise ImportError("Run: pip install easyocr") from e
        self.langs = langs or ["en"]
        self._reader = easyocr.Reader(self.langs, gpu=gpu)

    def read_bubble(self, image_bgr: np.ndarray, bubble: Bubble, padding: int = 6) -> str:
        masked = _masked_crop(image_bgr, bubble, padding)
        results = self._reader.readtext(masked, detail=0, paragraph=True)
        return " ".join(results).strip()

    def read_all(self, image_bgr: np.ndarray, bubbles: List[Bubble]) -> List[str]:
        return [self.read_bubble(image_bgr, b) for b in bubbles]


class MangaOCRBackend:
    """Japanese-only. Best accuracy on Japanese manga specifically."""

    def __init__(self):
        try:
            from manga_ocr import MangaOcr
        except ImportError as e:
            raise ImportError(
                "manga-ocr is not installed. Run: pip install manga-ocr\n"
                "(first run will also download model weights from Hugging Face)"
            ) from e
        self._model = MangaOcr()

    def read_bubble(self, image_bgr: np.ndarray, bubble: Bubble, padding: int = 6) -> str:
        masked = _masked_crop(image_bgr, bubble, padding)
        pil_img = Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        return self._model(pil_img).strip()

    def read_all(self, image_bgr: np.ndarray, bubbles: List[Bubble]) -> List[str]:
        return [self.read_bubble(image_bgr, b) for b in bubbles]


# Backwards-compatible alias (older versions of this script used this name)
MangaOCR = MangaOCRBackend
