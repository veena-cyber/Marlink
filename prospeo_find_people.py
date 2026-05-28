#!/usr/bin/env python3
"""
Find people at companies using the Prospeo Domain Search API.

Reads a CSV containing a column of company domains, queries Prospeo for the
people at each domain, keeps only those whose job title matches a target list,
and writes the matches to an output CSV.

Designed to run OUTSIDE the Claude Code web sandbox (which firewalls
api.prospeo.io). Run it on your own machine / a server where the API is
reachable.

Key features for large lists (e.g. millions of domains):
  - Checkpointing: every processed domain is recorded, so re-running resumes
    where it left off instead of re-spending credits.
  - Retries with exponential backoff on rate limits / transient errors.
  - Concurrency via a thread pool (configurable).
  - Streams input and appends output, so memory use stays flat regardless of
    input size.

Usage:
  export PROSPEO_API_KEY="pk_..."        # do NOT hardcode the key
  python3 prospeo_find_people.py \
      --input companies.csv \
      --domain-column domain \
      --output people.csv \
      --workers 4

Verify the request/response field names against the current Prospeo API docs
(https://prospeo.io/api/domain-search) before a large run — the endpoint shape
can change over time.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PROSPEO_DOMAIN_SEARCH_URL = "https://api.prospeo.io/domain-search"

# Default target titles. Matching is case-insensitive substring: a person is
# kept if any of these strings appears in their job title.
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
]

_write_lock = threading.Lock()
_checkpoint_lock = threading.Lock()


def title_matches(title, targets_lower):
    if not title:
        return False
    t = title.lower()
    return any(target in t for target in targets_lower)


def prospeo_domain_search(domain, api_key, limit, max_retries=4):
    """Call Prospeo Domain Search for one domain. Returns the list of people
    (raw dicts) or raises on unrecoverable error."""
    body = json.dumps({"company": domain, "limit": limit}).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-KEY": api_key}

    delay = 2.0
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            PROSPEO_DOMAIN_SEARCH_URL, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            # Prospeo wraps results as {"error": bool, "response": {...}}.
            if payload.get("error"):
                # Application-level error (e.g. no results) -> treat as empty.
                return []
            response = payload.get("response") or {}
            # Field name has historically been "email_list"; fall back to a few
            # likely alternatives so a minor schema change doesn't drop data.
            people = (
                response.get("email_list")
                or response.get("contacts")
                or response.get("people")
                or []
            )
            return people
        except urllib.error.HTTPError as e:
            # 429 = rate limited, 5xx = transient -> retry with backoff.
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            # 4xx (bad domain, no credits, auth) -> don't retry this domain.
            sys.stderr.write(f"[http {e.code}] {domain}: {e.reason}\n")
            return []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            sys.stderr.write(f"[error] {domain}: {e}\n")
            return []
    return []


def normalize_person(domain, p):
    """Map a Prospeo person record to our flat output row, tolerating
    different key spellings."""
    def pick(*keys):
        for k in keys:
            v = p.get(k)
            if v:
                return v
        return ""

    return {
        "domain": domain,
        "first_name": pick("first_name", "firstName"),
        "last_name": pick("last_name", "lastName"),
        "full_name": pick("full_name", "fullName", "name"),
        "job_title": pick("position", "job_title", "title", "jobTitle"),
        "seniority": pick("seniority"),
        "department": pick("department"),
        "email": pick("email"),
        "email_status": pick("email_status", "status", "verification"),
        "linkedin_url": pick("linkedin_url", "linkedin", "linkedinUrl"),
    }


def load_checkpoint(path):
    done = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = line.strip()
                if d:
                    done.add(d)
    return done


def append_checkpoint(path, domain):
    if not path:
        return
    with _checkpoint_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(domain + "\n")


def iter_domains(input_path, domain_column):
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if domain_column not in (reader.fieldnames or []):
            sys.exit(
                f"Column '{domain_column}' not found. "
                f"Available columns: {reader.fieldnames}"
            )
        seen = set()
        for row in reader:
            d = (row.get(domain_column) or "").strip().lower()
            # Normalize: strip scheme/path if a full URL slipped in.
            d = d.replace("https://", "").replace("http://", "").strip("/")
            if d and d not in seen:
                seen.add(d)
                yield d


def main():
    ap = argparse.ArgumentParser(description="Find people at companies via Prospeo.")
    ap.add_argument("--input", required=True, help="Input CSV with a domain column.")
    ap.add_argument("--domain-column", default="domain", help="Name of the domain column.")
    ap.add_argument("--output", default="prospeo_people.csv", help="Output CSV path.")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint file (default: <output>.done).")
    ap.add_argument("--limit", type=int, default=50,
                    help="Max people fetched per domain (1-100).")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent requests.")
    ap.add_argument("--no-filter", action="store_true",
                    help="Keep all returned people, ignore the title filter.")
    ap.add_argument("--titles-file", default=None,
                    help="Optional file with one target title per line "
                         "(overrides the built-in list).")
    args = ap.parse_args()

    api_key = os.environ.get("PROSPEO_API_KEY")
    if not api_key:
        sys.exit("Set the PROSPEO_API_KEY environment variable first.")

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

    def process(domain):
        people = prospeo_domain_search(domain, api_key, args.limit)
        rows = []
        for p in people:
            row = normalize_person(domain, p)
            if args.no_filter or title_matches(row["job_title"], targets_lower):
                rows.append(row)
        return domain, rows

    pending = (d for d in iter_domains(args.input, args.domain_column) if d not in done)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Submit in bounded batches so we don't queue millions of futures.
            batch = []
            futures = {}

            def drain(futures):
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
                    if stats["domains"] % 100 == 0:
                        sys.stderr.write(
                            f"  processed {stats['domains']} domains, "
                            f"{stats['matches']} matches\n"
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
        f"Done. Processed {stats['domains']} new domains, "
        f"wrote {stats['matches']} matching people to {args.output}\n"
    )


if __name__ == "__main__":
    main()
