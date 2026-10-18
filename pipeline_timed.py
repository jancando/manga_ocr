"""
pipeline.py

End-to-end: page image -> detect bubbles (via a manga/comic-specific
detector, classified by shape) -> OCR each bubble (SFX excluded because it's
a separate, non-bubble class from the detector) -> translate -> annotated
output image with a color legend + a JSON sidecar with all extracted/
translated text.

Usage:
    python pipeline.py --input path/to/page_or_folder --output out_dir
    python pipeline.py --input pages/ --output out_dir --langs ja --target en
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from bubble_detector import (
    detect_bubbles, find_local_bubble_contour, DetectorConfig, Bubble, LEGEND_COLORS
)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def draw_legend(img: np.ndarray) -> np.ndarray:
    """Appends a color-key strip along the bottom of the image."""
    h, w = img.shape[:2]
    bar_h = 50
    canvas = np.full((h + bar_h, w, 3), 255, dtype=np.uint8)
    canvas[:h] = img

    x = 15
    y = h + bar_h // 2
    for label, color in LEGEND_COLORS.items():
        cv2.rectangle(canvas, (x, y - 12), (x + 24, y + 12), color, -1)
        cv2.putText(canvas, label, (x + 32, y + 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        x += 32 + 12 * len(label) + 40
    return canvas


def annotate_page(img: np.ndarray, bubbles: List[Bubble]) -> np.ndarray:
    out = img.copy()
    for b in bubbles:
        color = LEGEND_COLORS.get(b.bubble_type, (128, 128, 128))
        cv2.drawContours(out, [b.contour], -1, color, 3)
        x, y, w, h = b.bbox
        label = f"#{b.id} {b.bubble_type}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(th + 10, y)  # keep label on-screen if bubble is near the top edge
        cv2.rectangle(out, (x, ty - th - 10), (x + tw + 8, ty), color, -1)
        cv2.putText(out, label, (x + 4, ty - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return draw_legend(out)


def _dedup_refined_bubbles(bubbles: List[Bubble], overlap_ratio_thresh: float = 0.6) -> List[Bubble]:
    """
    The comic detector can occasionally return two overlapping boxes for the
    same physical bubble -- e.g. one tight box around the full bubble and a
    second, smaller box around just part of its text (a stray fragment).
    These don't always share enough raw IoU to get caught by the detector's
    own NMS, but after refinement one box's area is almost entirely inside
    the other's. Catch that here: for any pair where the overlap covers most
    of the *smaller* box's area, keep only the larger one (more complete).
    """
    def area(b):
        return b.bbox[2] * b.bbox[3]

    def overlap_area(a, b):
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        ix0, iy0 = max(ax, bx), max(ay, by)
        ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        return max(0, ix1 - ix0) * max(0, iy1 - iy0)

    ordered = sorted(bubbles, key=area, reverse=True)
    kept: List[Bubble] = []
    for cand in ordered:
        is_dup = False
        for k in kept:
            ov = overlap_area(cand, k)
            if ov / area(cand) > overlap_ratio_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


def get_bubbles_via_comic_detector(img: np.ndarray, comic_detector,
                                    cfg: DetectorConfig, score_threshold: float = 0.3):
    """
    Locate bubbles with the manga/comic-specific detector (bbox only), then
    refine each into a precise shape/contour via a local classical-CV pass
    (bubble_detector.find_local_bubble_contour) -- this two-stage approach is
    what makes both steps reliable: the ML model handles the hard part
    (knowing where real bubbles vs. plain art are, on real full-color pages),
    while the local, tightly-cropped classical pass handles shape
    classification far better than it could on the whole page at once.

    Returns (bubbles, excluded_text_free) where excluded_text_free is the
    list of detected non-bubble text regions (SFX/captions) for visibility,
    not included in the main output.
    """
    detections = comic_detector.detect(img)
    bubble_dets = [d for d in detections if d.label == "bubble"]
    text_free_dets = [d for d in detections if d.label == "text_free"]

    bubbles = []
    for i, det in enumerate(bubble_dets):
        x0, y0, x1, y1 = det.bbox
        bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        b = find_local_bubble_contour(img, bbox, cfg=cfg)
        b.id = i
        bubbles.append(b)

    bubbles = _dedup_refined_bubbles(bubbles)

    # reading order: top row first, right-to-left (typical manga/webtoon layout)
    bubbles.sort(key=lambda b: (b.bbox[1] // 100, -b.bbox[0]))
    for new_id, b in enumerate(bubbles):
        b.id = new_id

    return bubbles, text_free_dets


def process_page(image_path: Path, output_dir: Path, ocr_engine, translator,
                 comic_detector=None, cfg: DetectorConfig = DetectorConfig()) -> dict:
    page_start = time.perf_counter()
    timings = {}

    # 1. Load image
    t0 = time.perf_counter()
    img = cv2.imread(str(image_path))
    timings["image_loading"] = time.perf_counter() - t0

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 2. Detect bubbles and refine their contours
    t0 = time.perf_counter()
    if comic_detector is not None:
        bubbles, excluded = get_bubbles_via_comic_detector(
            img, comic_detector, cfg
        )
    else:
        bubbles = detect_bubbles(img, cfg)
        excluded = []
    timings["bubble_detection_and_refinement"] = time.perf_counter() - t0

    records = []
    ocr_seconds = 0.0
    translation_seconds = 0.0
    ocr_bubble_count = 0
    translated_bubble_count = 0

    # 3. OCR and translation
    for b in bubbles:
        if ocr_engine:
            t0 = time.perf_counter()
            original_text = ocr_engine.read_bubble(img, b)
            elapsed = time.perf_counter() - t0
            ocr_seconds += elapsed
            ocr_bubble_count += 1
        else:
            original_text = ""

        translated_text = ""
        if translator and original_text:
            t0 = time.perf_counter()
            translated_text = translator.translate(original_text)
            translation_seconds += time.perf_counter() - t0
            translated_bubble_count += 1

        records.append({
            "id": b.id,
            "bubble_type": b.bubble_type,
            "confidence": round(b.confidence, 3),
            "bbox": {
                "x": b.bbox[0],
                "y": b.bbox[1],
                "w": b.bbox[2],
                "h": b.bbox[3],
            },
            "text_original": original_text,
            "text_translated": translated_text,
        })

    timings["ocr"] = ocr_seconds
    timings["translation"] = translation_seconds

    # 4. Annotate image
    t0 = time.perf_counter()
    annotated = annotate_page(img, bubbles)
    timings["annotation"] = time.perf_counter() - t0

    # 5. Save image and JSON
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    annotated_path = output_dir / f"{stem}_annotated.png"
    json_path = output_dir / f"{stem}.json"

    cv2.imwrite(str(annotated_path), annotated)

    # Save timings in the JSON sidecar too
    timings["saving_image_and_json"] = time.perf_counter() - t0
    timings["total_page_processing"] = time.perf_counter() - page_start

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": str(image_path),
            "bubbles": records,
            "excluded_non_bubble_text_count": len(excluded),
            "timings_seconds": {
                key: round(value, 4)
                for key, value in timings.items()
            },
        }, f, ensure_ascii=False, indent=2)

    return {
        "annotated_image": str(annotated_path),
        "data": str(json_path),
        "bubble_count": len(records),
        "excluded_count": len(excluded),
        "ocr_seconds": ocr_seconds,
        "ocr_bubble_count": ocr_bubble_count,
        "translation_seconds": translation_seconds,
        "translated_bubble_count": translated_bubble_count,
        "timings": timings,
    }


def main():
    parser = argparse.ArgumentParser(description="Manga OCR + translation + shape-legend pipeline")
    parser.add_argument("--input", required=True, help="Image file or folder of page images")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--detector", default="comic", choices=["comic", "classical"],
                         help="'comic' (default): ML model trained on manga/webtoon/comic pages "
                              "for locating bubbles -- much more reliable on real full-color art. "
                              "'classical': pure OpenCV heuristics, no extra download, but prone to "
                              "false positives on textured/full-color pages (works best on clean "
                              "black-and-white line art).")
    parser.add_argument("--backend", default="easyocr", choices=["easyocr", "manga-ocr","paddle-ocr"],
                         help="OCR engine: 'easyocr' (multi-language, default) or "
                              "'manga-ocr' (Japanese only, higher accuracy on Japanese)")
    parser.add_argument("--langs", default="en",
                         help="Comma-separated EasyOCR language codes, e.g. 'en', 'fr', 'ko', "
                              "'ch_sim,en'. Ignored when --backend=manga-ocr. Note: ja/ko/zh "
                              "each only combine with 'en', not with each other -- see README.")
    parser.add_argument("--target", default="en", help="Target translation language code (default: en)")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR (detection/legend only, for testing)")
    parser.add_argument("--no-translate", action="store_true", help="Skip translation step")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_dir():
        image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    else:
        image_paths = [input_path]

    comic_detector = None
    if args.detector == "comic":
        from comic_detector import ComicDetector
        comic_detector = ComicDetector()

    ocr_engine = None
    if not args.no_ocr:
        if args.backend == "manga-ocr":
            from ocr import MangaOCRBackend
            ocr_engine = MangaOCRBackend()
        elif args.backend == "paddle-ocr":
            from ocr import PaddleOCRBackend
            ocr_engine = PaddleOCRBackend()
        else:
            from ocr import EasyOCRBackend
            lang_list = [l.strip() for l in args.langs.split(",") if l.strip()]
            ocr_engine = EasyOCRBackend(langs=lang_list)

    translator = None
    if not args.no_translate:
        from translator import GoogleTranslatorBackend
        translator = GoogleTranslatorBackend(source="auto", target=args.target)

    for image_path in image_paths:
        result = process_page(image_path, output_dir, ocr_engine, translator, comic_detector)
        print(f"{image_path.name}: {result['bubble_count']} bubbles "
              f"({result['excluded_count']} non-bubble text regions excluded) "
              f"-> {result['annotated_image']}")

        if result["ocr_bubble_count"]:
            avg_ocr = result["ocr_seconds"] / result["ocr_bubble_count"]
            print(f"  OCR time: {result['ocr_seconds']:.2f}s total "
                  f"({avg_ocr:.3f}s/bubble, {result['ocr_bubble_count']} bubbles)")
        elif args.no_ocr:
            print("  OCR time: skipped (--no-ocr)")
        else:
            print("  OCR time: 0.00s (no bubbles detected)")
        
        print("  Timing breakdown:")
        for stage, seconds in result["timings"].items():
            print(f"    {stage.replace('_', ' ').title():32s} "
                  f"{seconds:8.3f}s")

if __name__ == "__main__":
    main()
