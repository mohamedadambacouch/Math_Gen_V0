"""
ingest_process/extract.py
=========================
Extract structured math questions from exam paper images or PDFs.

Pipeline
--------
1. Input  : folder of images  OR  one or more PDF files
2. MinerU : two_step_extract() → structured [{type, content/bbox}, ...]
3. Qwen2-VL : describes any embedded graph/diagram regions
4. Parser : assembles question dicts  { question_id, question, choices,
                                        has_image, image_description }
5. Output : JSON file (default  final_dataset.json), merged with any
            existing content so re-runs are safe

Usage
-----
  # images in a folder
  python ingest_process/extract.py --input path/to/images

  # one or more PDFs
  python ingest_process/extract.py --input paper1.pdf paper2.pdf

  # custom output file
  python ingest_process/extract.py --input path/to/images --output my_out.json

  # higher DPI for better OCR quality on PDFs (default 200)
  python ingest_process/extract.py --input paper.pdf --dpi 300
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient

# ── Constants ──────────────────────────────────────────────────────────────────
MINERU_MODEL_ID = "opendatalab/MinerU2.5-2509-1.2B"
QWEN_MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_EXTS      = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# ── Model loading (done once) ──────────────────────────────────────────────────

def load_models():
    print("Loading MinerU model …")
    mineru_processor = AutoProcessor.from_pretrained(MINERU_MODEL_ID)
    mineru_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MINERU_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    client = MinerUClient(
        model=mineru_model,
        processor=mineru_processor,
        backend="transformers",
    )
    print("  ✅ MinerU ready")

    print("Loading Qwen2-VL model …")
    qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("  ✅ Qwen2-VL ready")

    return client, qwen_processor, qwen_model


# ── PDF → images ───────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
    """Convert every page of a PDF to a PIL RGB image."""
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    images = []
    zoom = dpi / 72  # 72 is fitz's default DPI
    mat  = fitz.Matrix(zoom, zoom)
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        print(f"  converted page {page_num}/{len(doc)}")
    doc.close()
    return images


# ── Collect input images ───────────────────────────────────────────────────────

def collect_inputs(inputs: List[str], dpi: int) -> List[Dict[str, Any]]:
    """
    Returns a flat list of dicts:
        { "name": str, "image": PIL.Image }
    Handles image files, image folders, and PDFs.
    """
    items = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files = sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            )
            if not files:
                print(f"⚠  No images found in {p}")
            for f in files:
                items.append({
                    "name": f.name,
                    "image": Image.open(f).convert("RGB"),
                })
                print(f"  queued image: {f.name}")
        elif p.suffix.lower() == ".pdf":
            print(f"Converting PDF: {p.name}")
            pages = pdf_to_images(p, dpi=dpi)
            for i, img in enumerate(pages, start=1):
                items.append({
                    "name": f"{p.stem}_page{i:03d}.png",
                    "image": img,
                })
        elif p.suffix.lower() in IMAGE_EXTS:
            items.append({
                "name": p.name,
                "image": Image.open(p).convert("RGB"),
            })
            print(f"  queued image: {p.name}")
        else:
            print(f"⚠  Skipping unsupported file: {p}")
    return items


# ── Qwen2-VL image description ─────────────────────────────────────────────────

def describe_image(
    image: Image.Image,
    qwen_processor: AutoProcessor,
    qwen_model: Qwen2VLForConditionalGeneration,
    max_new_tokens: int = 200,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Describe this graph or diagram precisely. "
                        "Include all visible numbers, labels, and comparisons. "
                        "Be concise and factual."
                    ),
                },
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(qwen_model.device)

    with torch.no_grad():
        output = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)

    return qwen_processor.batch_decode(output, skip_special_tokens=True)[0]


# ── Crop helper ────────────────────────────────────────────────────────────────

def crop_bbox(img: Image.Image, bbox: List[float]) -> Image.Image:
    """bbox is [x0, y0, x1, y1] in 0-1 relative coords."""
    w, h = img.size
    return img.crop((
        int(bbox[0] * w),
        int(bbox[1] * h),
        int(bbox[2] * w),
        int(bbox[3] * h),
    ))


# ── Choice normaliser ──────────────────────────────────────────────────────────

def clean_choice(text: str) -> str:
    m = re.match(r"^([A-E])\s*(.*)", text)
    if m:
        return f"{m.group(1)}. {m.group(2).strip()}"
    return text


# ── MinerU output parser ───────────────────────────────────────────────────────

def parse_mineru_output(
    output: List[Dict],
    source_image: Image.Image,
    qwen_processor: AutoProcessor,
    qwen_model: Qwen2VLForConditionalGeneration,
) -> List[Dict]:
    """
    Walk the MinerU structured output and assemble question dicts.
    Questions start with a leading digit (e.g. "1", "2 What is…").
    Choices start with A-E.
    Image regions are cropped and described with Qwen2-VL.
    """
    questions: List[Dict] = []
    current_q: Optional[Dict] = None

    for item in output:
        if item["type"] == "text":
            text = item["content"].strip()
            if not text:
                continue

            # Detect question start: "1", "1 Some text", "12 Something"
            q_match = re.match(r"^(\d+)\s*(.*)", text)
            if q_match:
                if current_q:
                    questions.append(current_q)
                current_q = {
                    "question_id": int(q_match.group(1)),
                    "question":    q_match.group(2).strip(),
                    "choices":     [],
                    "has_image":   False,
                    "image_description": None,
                }

            elif current_q is not None:
                if re.match(r"^[A-E]", text):
                    current_q["choices"].append(clean_choice(text))
                else:
                    # continuation of question text
                    current_q["question"] += " " + text

        elif item["type"] == "image" and current_q is not None:
            if current_q["image_description"] is None:
                current_q["has_image"] = True
                bbox = item.get("bbox")
                if bbox:
                    region = crop_bbox(source_image, bbox)
                else:
                    region = source_image
                try:
                    desc = describe_image(region, qwen_processor, qwen_model)
                    current_q["image_description"] = desc
                except Exception as e:
                    current_q["image_description"] = f"[description failed: {e}]"

    if current_q:
        questions.append(current_q)

    return questions


# ── Dedup helper ───────────────────────────────────────────────────────────────

def load_existing(output_path: Path) -> Dict[str, Dict]:
    """Return {question_text: record} for all entries already in the output file."""
    if not output_path.exists():
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {entry["question"].strip(): entry for entry in data}
    except Exception:
        return {}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract math questions from images or PDFs using MinerU + Qwen2-VL"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="One or more image files, image folders, or PDF files",
    )
    parser.add_argument(
        "--output", "-o",
        default="final_dataset.json",
        help="Output JSON file (default: final_dataset.json)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF-to-image conversion (default: 200)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # ── Load existing results (for dedup + merge) ──────────────────────────────
    existing = load_existing(output_path)
    print(f"Existing questions in {output_path.name}: {len(existing)}")

    # ── Collect all inputs ─────────────────────────────────────────────────────
    print("\nCollecting inputs …")
    items = collect_inputs(args.input, dpi=args.dpi)
    if not items:
        print("No valid inputs found. Exiting.")
        sys.exit(1)
    print(f"Total pages/images to process: {len(items)}\n")

    # ── Load models ────────────────────────────────────────────────────────────
    client, qwen_processor, qwen_model = load_models()

    # ── Process each image ─────────────────────────────────────────────────────
    new_questions: List[Dict] = []
    skipped = 0

    for idx, item in enumerate(items, start=1):
        name  = item["name"]
        image = item["image"]
        print(f"\n[{idx}/{len(items)}] Processing: {name}")

        try:
            raw_output = client.two_step_extract(image=image)
        except Exception as e:
            print(f"  ❌ MinerU failed: {e}")
            continue

        parsed = parse_mineru_output(raw_output, image, qwen_processor, qwen_model)
        print(f"  found {len(parsed)} question(s)")

        for q in parsed:
            q_text = q["question"].strip()
            if q_text in existing:
                skipped += 1
                print(f"  ↩  skipped duplicate: {q_text[:60]}…")
            else:
                existing[q_text] = q
                new_questions.append(q)
                print(f"  ✅ Q{q['question_id']}: {q_text[:60]}")

    # ── Save merged results ────────────────────────────────────────────────────
    all_questions = list(existing.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"\n{'─'*50}")
    print(f"New questions extracted : {len(new_questions)}")
    print(f"Duplicates skipped      : {skipped}")
    print(f"Total in {output_path.name:<20}: {len(all_questions)}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
