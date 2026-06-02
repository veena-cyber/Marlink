#!/usr/bin/env python3
"""
Find people at companies using the Clay People Search API.

Reads a CSV containing a column of company domains, queries Clay for contacts
at each domain filtered by job title keywords, and writes matches to an output CSV.

Designed to run on your LOCAL machine (not the Claude Code sandbox).

Key features for large lists (e.g. 6M domains):
  - Checkpointing: every processed domain is recorded so re-running resumes
    where it left off without re-spending credits.
  - Retries with exponential backoff on rate limits / transient errors.
  - Concurrency via a thread pool (configurable, keep low to avoid rate limits).
  - Streams input and appends output so memory stays flat regardless of input size.

Setup:
  pip install requests   # only external dependency

Usage:
  export CLAY_API_KEY="clay_..."         # do NOT hardcode the key
  python3 clay_find_people.py \
      --input companies.csv \
      --domain-column domain \
      --output clay_people.csv \
      --workers 2

Verify the endpoint URL and request/response field names against the current
Clay API docs (https://docs.clay.com) before a large run — the schema can change.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Clay people-search endpoint — verify against https://docs.clay.com
CLAY_PEOPLE_SEARCH_URL = "https://api.clay.com/v1/sources/people-search"

# Job title keywords to keep (case-insensitive substring match).
# A person is included if ANY of these strings appears in their job title.
DEFAULT_TARGET_TITLES = [
    "Head of Retention", "VP of Retention", "Director of Retention",
    "Director of Retention Marketing", "Retention Marketing Manager",
    "Head of Lifecycle Marketing", "Director of Lifecycle Marketing",
    "Lifecycle Marketing Manager", "Head of Subscriptions",
    "Head of Customer Lifecycle", "Director of CRM", "Head of CRM",
    "CRM Manager", "Email Marketing Manager", "Chief Marketing Officer",
    "CMO", "Chief Revenue Officer", "CRO", "Founder", "Co-Founder", "CEO",
    "VP of Marketing", "VP of Growth", "VP of Ecommerce",
    "VP of Customer Success", "VP of Performance Marketing", "Head of Growth",
    "Head of Ecommerce", "Head of Marketing", "Director of Ecommerce",
    "Director of Growth", "Director of E-commerce", "Head of Data Analytics",
    "Head of Business Intelligence", "Head of Customer Analytics",
    "Director of Marketing Analytics", "Director of Customer Analytics",
    "Director of Data", "Marketing Analytics Manager",
    "Customer Analytics Manager", "Customer Insights Manager",
    "Head of Revenue Operations", "Director of Sales Operations",
    "Director of Strategic Analytics", "Head of Customer Experience",
]

OUTPUT_FIELDS = [
    "domain", "first_name", "last_name", "full_name", "job_title",
    "seniority", "department", "email", "email_status", "linkedin_url",
    "location", "company_name",
]

_write_lock = threading.Lock()
_checkpoint_lock = threading.Lock()


def title_matches(title: str, targets_lower: list[str]) -> bool:
    if not title:
        return False
    t = title.lower()
    return any(target in t for target in targets_lower)


def clay_search(domain: str, api_key: str, limit: int, title_keywords: list[str], max_retries: int = 4) -> list[dict]:
    """
    Call Clay People Search for one domain. Returns list of raw contact dicts.

    Clay request body shape (verify against docs):
      {
        "domain": "example.com",
        "limit": 25,
        "filters": {
          "job_title_keywords": ["CEO", "Founder", ...]
        }
      }
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                CLAY_PEOPLE_SEARCH_URL,
                json={"domain": domain, "limit": limit, "filters": {"job_title_keywords": title_keywords}},
                headers=headers,
                timeout=60,
                verify=False,
            )
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            if not resp.ok:
                sys.stderr.write(f"[http {resp.status_code}] {domain}: {resp.text[:200]}\n")
                return []
            payload = resp.json()
            contacts = (
                payload.get("contacts")
                or payload.get("people")
                or payload.get("results")
                or payload.get("data")
                or []
            )
            return contacts
        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            sys.stderr.write(f"[error] {domain}: {e}\n")
            return []
    return []


def normalize_person(domain: str, p: dict) -> dict:
    """Map a Clay contact record to our flat output row."""
    def pick(*keys):
        for k in keys:
            v = p.get(k)
            if v:
                return str(v)
        return ""

    return {
        "domain": domain,
        "first_name": pick("first_name", "firstName"),
        "last_name": pick("last_name", "lastName"),
        "full_name": pick("full_name", "fullName", "name"),
        "job_title": pick("job_title", "title", "position", "jobTitle"),
        "seniority": pick("seniority"),
        "department": pick("department"),
        "email": pick("email", "work_email", "workEmail"),
        "email_status": pick("email_status", "emailStatus", "verification"),
        "linkedin_url": pick("linkedin_url", "linkedin", "linkedinUrl"),
        "location": pick("location", "city", "country"),
        "company_name": pick("company_name", "company", "companyName", "organization"),
    }


def load_checkpoint(path: str) -> set[str]:
    done: set[str] = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = line.strip()
                if d:
                    done.add(d)
    return done


def append_checkpoint(path: str, domain: str) -> None:
    if not path:
        return
    with _checkpoint_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(domain + "\n")


def iter_domains(input_path: str, domain_column: str):
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if domain_column not in (reader.fieldnames or []):
            sys.exit(
                f"Column '{domain_column}' not found. "
                f"Available columns: {reader.fieldnames}"
            )
        seen: set[str] = set()
        for row in reader:
            d = (row.get(domain_column) or "").strip().lower()
            d = d.replace("https://", "").replace("http://", "").strip("/")
            if d and d not in seen:
                seen.add(d)
                yield d


def main() -> None:
    ap = argparse.ArgumentParser(description="Find people at companies via Clay API.")
    ap.add_argument("--input", required=True, help="Input CSV with a domain column.")
    ap.add_argument("--domain-column", default="domain", help="Name of the domain column (default: domain).")
    ap.add_argument("--output", default="clay_people.csv", help="Output CSV path.")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint file path (default: <output>.done).")
    ap.add_argument("--limit", type=int, default=25,
                    help="Max contacts fetched per domain (default: 25).")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent requests (default: 2; keep low to avoid rate limits).")
    ap.add_argument("--no-filter", action="store_true",
                    help="Keep all returned contacts, ignore the title filter.")
    ap.add_argument("--titles-file", default=None,
                    help="File with one target title per line (overrides built-in list).")
    args = ap.parse_args()

    api_key = os.environ.get("CLAY_API_KEY")
    if not api_key:
        sys.exit("Set the CLAY_API_KEY environment variable first.\n  export CLAY_API_KEY='clay_...'")

    if args.titles_file:
        with open(args.titles_file, "r", encoding="utf-8") as f:
            targets = [ln.strip() for ln in f if ln.strip()]
    else:
        targets = DEFAULT_TARGET_TITLES
    targets_lower = [t.lower() for t in targets]

    checkpoint_path = args.checkpoint or (args.output + ".done")
    done = load_checkpoint(checkpoint_path)
    if done:
        sys.stderr.write(f"Resuming: {len(done)} domains already processed.\n")

    output_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    out_f = open(args.output, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
    if not output_exists:
        writer.writeheader()
        out_f.flush()

    stats = {"domains": 0, "matches": 0}

    def process(domain: str) -> tuple[str, list[dict]]:
        people = clay_search(domain, api_key, args.limit, targets, max_retries=4)
        rows = []
        for p in people:
            row = normalize_person(domain, p)
            if args.no_filter or title_matches(row["job_title"], targets_lower):
                rows.append(row)
        return domain, rows

    pending = (d for d in iter_domains(args.input, args.domain_column) if d not in done)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            batch: list[str] = []

            def drain(futures: dict) -> None:
                for fut in as_completed(futures):
                    domain, rows = fut.result()
                    with _write_lock:
                        for row in rows:
                            writer.writerow(row)
                        if rows:
                            out_f.flush()
                    append_checkpoint(checkpoint_path, domain)
                    stats["domains"] += 1
                    stats["matches"] += len(rows)
                    if stats["domains"] % 500 == 0:
                        sys.stderr.write(
                            f"  processed {stats['domains']:,} domains, "
                            f"{stats['matches']:,} matches so far\n"
                        )

            for domain in pending:
                batch.append(domain)
                if len(batch) >= args.workers * 8:
                    futures = {pool.submit(process, d): d for d in batch}
                    drain(futures)
                    batch = []
            if batch:
                futures = {pool.submit(process, d): d for d in batch}
                drain(futures)
    finally:
        out_f.close()

    sys.stderr.write(
        f"\nDone. Processed {stats['domains']:,} new domains, "
        f"wrote {stats['matches']:,} matching people to {args.output}\n"
    )


if __name__ == "__main__":
    main()
