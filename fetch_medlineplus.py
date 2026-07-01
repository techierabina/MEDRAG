# fetch_medlineplus.py
# This is how we get real medical content into MedRAG — not by scraping
# random websites (legally messy and inconsistent quality), but by pulling
# from MedlinePlus, the National Library of Medicine's free public API.
# It's built for exactly this: no API key, no scraping, no rate-limit tricks.
# We just ask it for a topic by name and it hands back a clean summary.
#
# Run this once (or whenever you want to grow the corpus) BEFORE ingest.py.
# It writes one .txt file per topic into data/sample_docs/.

import os
import re
import time
import requests
import xml.etree.ElementTree as ET

# The official MedlinePlus search endpoint. db=healthTopics searches
# consumer health topic summaries — exactly the clinical-QA content we want.
BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"

OUTPUT_DIR = "./data/sample_docs"

# A starter set of common clinical topics. Add/remove freely —
# each one becomes its own MedlinePlus lookup and its own .txt file.
TOPICS = [
    "type 2 diabetes", "type 1 diabetes", "hypertension", "asthma",
    "chronic obstructive pulmonary disease", "coronary artery disease",
    "heart failure", "atrial fibrillation", "stroke", "migraine",
    "epilepsy", "depression", "anxiety disorders", "bipolar disorder",
    "chronic kidney disease", "urinary tract infection", "pneumonia",
    "tuberculosis", "influenza", "common cold", "covid-19",
    "hepatitis B", "hepatitis C", "cirrhosis", "gastroesophageal reflux disease",
    "peptic ulcer", "irritable bowel syndrome", "celiac disease",
    "crohn's disease", "ulcerative colitis", "rheumatoid arthritis",
    "osteoarthritis", "osteoporosis", "gout", "lupus",
    "hypothyroidism", "hyperthyroidism", "anemia", "sickle cell disease",
    "deep vein thrombosis", "pulmonary embolism", "psoriasis", "eczema",
    "acne", "skin cancer", "breast cancer", "lung cancer",
    "colorectal cancer", "prostate cancer", "obesity", "high cholesterol",
    "sleep apnea", "insomnia", "allergies", "sinusitis",
]


def clean_html(raw_html: str) -> str:
    # MedlinePlus summaries come with light HTML markup (<p>, <a>, etc.)
    # We just want plain text for chunking/embedding, so strip tags
    # and collapse extra whitespace left behind.
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_topic(term: str) -> dict | None:
    # Ask MedlinePlus for the top match on this term.
    params = {"db": "healthTopics", "term": term, "retmax": 1}
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    doc = root.find(".//document")
    if doc is None:
        print(f"  No result for '{term}' — skipping.")
        return None

    title, summary = None, None
    for content in doc.findall("content"):
        name = content.get("name")
        if name == "title":
            title = clean_html("".join(content.itertext()))
        elif name == "FullSummary":
            summary = clean_html("".join(content.itertext()))

    if not title or not summary:
        print(f"  Incomplete record for '{term}' — skipping.")
        return None

    return {"title": title, "summary": summary, "url": doc.get("url", "")}


def save_topic(topic: dict):
    # Filenames should be filesystem-safe and predictable.
    safe_name = re.sub(r"[^a-z0-9]+", "_", topic["title"].lower()).strip("_")
    filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{topic['title']}\n\n")
        f.write(topic["summary"])
        f.write(f"\n\nSource: MedlinePlus.gov — {topic['url']}")

    print(f"  Saved: {filepath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== Fetching {len(TOPICS)} topics from MedlinePlus ===\n")

    saved = 0
    for term in TOPICS:
        try:
            topic = fetch_topic(term)
            if topic:
                save_topic(topic)
                saved += 1
        except requests.RequestException as e:
            print(f"  Error fetching '{term}': {e}")

        # MedlinePlus caps at 85 requests/minute per IP — this keeps us
        # well under that so we never get throttled.
        time.sleep(0.5)

    print(f"\n✅ Done! {saved}/{len(TOPICS)} topics saved to {OUTPUT_DIR}")
    print("Next: run `python ingest.py` to embed and store them.")


if __name__ == "__main__":
    main()