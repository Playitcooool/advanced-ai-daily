#!/usr/bin/env python3
"""
Fetch the latest AI/ML papers from arXiv and identify trending topics.
This script runs as part of the GitHub Actions daily update workflow.

It searches multiple categories and extracts the most discussed concepts
for generating tutorial content.
"""

import urllib.request
import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime, timedelta

ARXIV_API = "https://export.arxiv.org/api/query"
OUTPUT_DIR = "scripts"

CATEGORIES = [
    "cs.LG",   # ML
    "cs.CL",   # NLP  
    "cs.AI",   # AI
    "cs.CV",   # Vision
    "stat.ML", # Stats ML
]

HOT_KEYWORDS = [
    "reinforcement learning", "GRPO", "PPO", "DPO",
    "mixture of experts", "MoE",
    "speculative decoding", "speculative",
    "reasoning", "chain of thought", "tree of thought",
    "agent", "multi-agent", "tool use",
    "test-time compute", "inference-time scaling",
    "long context", "KV cache",
    "alignment", "RLHF",
    "diffusion", "flow matching",
]

def fetch_arxiv(search_query, max_results=10, sort_by="submittedDate"):
    """Fetch papers from arXiv API."""
    url = f"{ARXIV_API}?search_query={search_query}&sortBy={sort_by}&sortOrder=descending&max_results={max_results}"
    
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "AdvancedAIDaily/1.0 (tutorial generator)")
    
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()

def parse_arxiv_response(xml_data):
    """Parse arXiv XML response into a list of paper dicts."""
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    root = ET.fromstring(xml_data)
    
    papers = []
    for entry in root.findall("a:entry", ns):
        title = entry.find("a:title", ns).text.strip().replace("\n", " ")
        arxiv_id = entry.find("a:id", ns).text.strip().split("/abs/")[-1]
        published = entry.find("a:published", ns).text[:10]
        summary = entry.find("a:summary", ns).text.strip()
        authors = ", ".join(
            a.find("a:name", ns).text 
            for a in entry.findall("a:author", ns)
        )
        categories = [
            c.get("term") 
            for c in entry.findall("a:category", ns)
        ]
        
        papers.append({
            "title": title,
            "arxiv_id": arxiv_id,
            "published": published,
            "summary": summary,
            "authors": authors,
            "categories": categories,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
        })
    
    return papers

def score_relevance(papers):
    """Score papers by relevance to hot keywords."""
    scored = []
    for paper in papers:
        text = (paper["title"] + " " + paper["summary"]).lower()
        score = sum(
            1 for kw in HOT_KEYWORDS 
            if kw.lower() in text
        )
        scored.append((*paper.items(),))
    
    return scored

def main():
    all_papers = []
    seen_ids = set()
    
    # Fetch from each category
    for cat in CATEGORIES:
        try:
            xml_data = fetch_arxiv(f"cat:{cat}", max_results=5)
            papers = parse_arxiv_response(xml_data)
            
            for p in papers:
                if p["arxiv_id"] not in seen_ids:
                    all_papers.append(p)
                    seen_ids.add(p["arxiv_id"])
        except Exception as e:
            print(f"Warning: Failed to fetch {cat}: {e}")
    
    # Also search hot topics directly
    for keyword in ["LLM", "reinforcement learning", "agent"]:
        try:
            xml_data = fetch_arxiv(f"all:{keyword}", max_results=5, sort_by="relevance")
            papers = parse_arxiv_response(xml_data)
            for p in papers:
                if p["arxiv_id"] not in seen_ids:
                    all_papers.append(p)
                    seen_ids.add(p["arxiv_id"])
        except Exception:
            pass
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "daily_papers.json")
    with open(output_path, "w") as f:
        json.dump({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "paper_count": len(all_papers),
            "papers": all_papers,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Fetched {len(all_papers)} unique papers")
    print(f"Saved to {output_path}")
    
    # Output for GitHub Actions
    titles = [p["title"] for p in all_papers[:5]]
    print("\nTop papers today:")
    for t in titles:
        print(f"  - {t}")

if __name__ == "__main__":
    main()
