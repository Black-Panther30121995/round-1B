import fitz  # PyMuPDF
import json
import sys
from pathlib import Path
import re
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Setup
logging.basicConfig(level=logging.WARNING)
model = model = SentenceTransformer("/root/.cache/torch/sentence_transformers/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2")


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def is_heading(text, size, threshold):
    text = clean(text)
    return (
        len(text) > 0
        and len(text.split()) <= 10
        and (text[0].isupper() or text[0].isdigit())
        and size >= threshold
        and text.lower() not in {"table of contents", "index", "references", "appendix"}
    )

def extract_sections(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Could not open {pdf_path}: {e}")
        return []

    all_sections = []
    font_sizes = [
        span["size"]
        for page in doc
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]

    if not font_sizes:
        return []

    threshold = np.percentile(font_sizes, 75)
    current_section = None
    buffer = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            lines = block.get("lines", [])
            block_text = clean(" ".join(span["text"] for line in lines for span in line.get("spans", [])))
            if not block_text:
                continue

            sizes = [span["size"] for line in lines for span in line.get("spans", [])]
            max_size = max(sizes) if sizes else 0

            if is_heading(block_text, max_size, threshold):
                if current_section and buffer:
                    current_section["subsections"].append({
                        "text": clean(" ".join(buffer)),
                        "page": current_section["page"]
                    })
                current_section = {
                    "text": block_text,
                    "page": page_num + 1,
                    "subsections": []
                }
                all_sections.append(current_section)
                buffer = []
            elif current_section:
                buffer.append(block_text)

        if current_section and buffer:
            current_section["subsections"].append({
                "text": clean(" ".join(buffer)),
                "page": current_section["page"]
            })
            buffer = []

    doc.close()
    return all_sections

def score_texts(texts, context_emb):
    embs = model.encode([clean(t) for t in texts], convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(embs, context_emb).cpu().numpy().flatten()
    return sims.tolist()

def process_single_pdf(args):
    pdf_path, context_emb = args
    try:
        secs = extract_sections(pdf_path)
        pdf_name = pdf_path.name
        result = []
        for sec in secs:
            sec["document"] = pdf_name
            result.append((sec, sec["text"]))
            for sub in sec.get("subsections", []):
                sub["document"] = pdf_name
                result.append((sub, sub["text"]))
        return result
    except Exception as e:
        logging.error(f"Error in {pdf_path.name}: {e}")
        return []

def process_documents(pdf_paths, persona, job):
    context = f"{persona} {job}"
    context_emb = model.encode(context, convert_to_tensor=True)

    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_pdf, [(p, context_emb) for p in pdf_paths])

    all_secs, all_texts, meta_refs = [], [], []
    for r in results:
        for meta, text in r:
            if "subsections" in meta:
                all_secs.append(meta)
            all_texts.append(text)
            meta_refs.append(meta)

    scores = score_texts(all_texts, context_emb)
    for meta, score in zip(meta_refs, scores):
        meta["importance"] = float(score)
        meta["refined_text"] = clean(meta["text"][:1000] + "..." if len(meta["text"]) > 1000 else meta["text"])

    top_sections = sorted([s for s in all_secs if s.get("importance", 0) > 0.1],
                          key=lambda x: x["importance"], reverse=True)[:5]
    for i, sec in enumerate(top_sections):
        sec["rank"] = i + 1

    top_subs = sorted(
        [sub for sec in all_secs for sub in sec.get("subsections", []) if sub.get("importance", 0) > 0.1],
        key=lambda x: x["importance"], reverse=True)[:5]
    for i, sub in enumerate(top_subs):
        sub["rank"] = i + 1

    return top_sections, top_subs

def save_output(sections, subsections, input_data, output_file):
    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_data["documents"]],
            "persona": input_data["persona"]["role"],
            "job_to_be_done": input_data["job_to_be_done"]["task"],
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [
            {
                "document": sec["document"],
                "section_title": sec["text"],
                "importance_rank": sec["rank"],
                "page_number": sec["page"]
            } for sec in sections
        ],
        "subsection_analysis": [
            {
                "document": sub["document"],
                "refined_text": sub["refined_text"],
                "page_number": sub["page"],
                "importance_rank": sub["rank"]
            } for sub in subsections
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <input_json_path> <input_pdfs_dir> <output_json_path>")
        sys.exit(1)

    input_json = Path(sys.argv[1])
    input_pdfs = Path(sys.argv[2])
    output_json = Path(sys.argv[3])

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    pdf_paths = [input_pdfs / d["filename"] for d in data["documents"] if (input_pdfs / d["filename"]).exists()]
    if not pdf_paths:
        logging.error("No valid PDFs found.")
        return

    sections, subsections = process_documents(pdf_paths, data["persona"]["role"], data["job_to_be_done"]["task"])
    save_output(sections, subsections, data, output_json)

if __name__ == "__main__":
    main()
