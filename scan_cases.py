#!/usr/bin/env python3
"""Weekly scanner for NYC felony-assault DA press releases.

Reads cases-2026.json, scrapes each DA's press-release listing page for
releases posted since the last scan, asks Claude Haiku to classify and
extract fields, and appends qualifying cases to cases-2026.json.

Idempotent: keys on source_url. A URL already in the file is skipped.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
from anthropic import Anthropic
from bs4 import BeautifulSoup


def _ensure_pypdf() -> None:
    """Install pypdf at runtime if it isn't already installed.

    The Bronx DA publishes press releases as PDFs, so we need a PDF text
    extractor. Ideally pypdf would be in the GitHub Actions workflow's
    pip-install line (alongside requests/beautifulsoup4/lxml/anthropic),
    but that file lives under .github/workflows/ and editing it requires
    the OAuth `workflow` scope, which the bot account doesn't carry.

    Doing the install here keeps the runtime self-bootstrapping. It costs
    ~5 seconds on first run; pypdf is tiny (no native deps).
    """
    try:
        import pypdf  # noqa: F401
    except ImportError:
        import subprocess
        print("[setup] installing pypdf for Bronx PDF extraction…", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "pypdf"]
        )


_ensure_pypdf()

HERE = Path(__file__).resolve().parent.parent
CASES_PATH = HERE / "cases-2026.json"
YEAR = 2026
# Only look back this many days if the file has no last_scan_date.
DEFAULT_LOOKBACK_DAYS = 14
# Hard cap on releases fetched per DA per run to avoid runaway API usage.
MAX_RELEASES_PER_DA = 30

USER_AGENT = (
    "NYC-Assault-Tracker/1.0 (+https://vitalcity-nyc.github.io/nyc-assault-tracker/)"
)


@dataclass
class Release:
    url: str
    title: str
    posted: date | None
    borough_hint: str  # DA borough, used as a default if Haiku can't tell


# ---------- Per-DA listing scrapers ----------
#
# As of April 2026 every NYC DA except the Bronx publishes a working
# WordPress RSS feed. RSS gives us {title, link, pubDate} cleanly without
# fragile HTML selectors that break each time a designer updates the site.
# The Bronx DA is still static HTML linking to PDFs, so we keep an HTML
# scraper for them and pull the body via pypdf when classifying.


def fetch(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.content


def parse_date(s: str) -> date | None:
    s = s.strip()
    # Common shapes: "January 2, 2026", "Jan 2, 2026", "2026-01-02", "01/02/2026"
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%B %d %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def parse_rss_pubdate(s: str) -> date | None:
    # RSS pubDate is RFC-822: "Fri, 24 Apr 2026 16:59:05 +0000"
    if not s:
        return None
    try:
        return datetime.strptime(s.strip()[:25], "%a, %d %b %Y %H:%M:%S").date()
    except ValueError:
        return None


def scrape_rss(feed_url: str, borough_hint: str, log_name: str) -> list[Release]:
    out: list[Release] = []
    try:
        xml = fetch(feed_url)
    except Exception as e:
        print(f"[{log_name}] feed fetch failed: {e}", file=sys.stderr)
        return out
    # Lightweight regex parsing — feed payloads are small and well-formed enough
    # that pulling in feedparser as a third dependency isn't worth it.
    items = re.findall(r"<item>(.*?)</item>", xml, re.DOTALL)
    for raw in items[:MAX_RELEASES_PER_DA]:
        title_m = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", raw, re.DOTALL)
        link_m = re.search(r"<link>([^<]+)</link>", raw)
        pub_m = re.search(r"<pubDate>([^<]+)</pubDate>", raw)
        if not title_m or not link_m:
            continue
        title = re.sub(r"\s+", " ", title_m.group(1)).strip()
        href = link_m.group(1).strip()
        posted = parse_rss_pubdate(pub_m.group(1)) if pub_m else None
        out.append(Release(url=href, title=title, posted=posted, borough_hint=borough_hint))
    return out


def scrape_manhattan() -> list[Release]:
    return scrape_rss(
        "https://manhattanda.org/category/news/press-release/feed/",
        "Manhattan",
        "manhattan",
    )


def scrape_brooklyn() -> list[Release]:
    return scrape_rss(
        "https://www.brooklynda.org/feed/",
        "Brooklyn",
        "brooklyn",
    )


def scrape_queens() -> list[Release]:
    return scrape_rss(
        "https://queensda.org/category/press-releases/feed/",
        "Queens",
        "queens",
    )


def scrape_staten_island() -> list[Release]:
    return scrape_rss(
        "https://statenislandda.org/news/feed/",
        "Staten Island",
        "staten_island",
    )


def scrape_bronx() -> list[Release]:
    """Bronx DA still publishes HTML linking to dated PDF press releases.
    The PDFs live at /downloads/pdf/pr/<year>/<n>-<year>-<slug>.pdf — the file
    name is the most reliable date signal because some PDFs lack any header
    date. We try the URL first, then fall back to the file's mtime if needed.
    """
    url = "https://www.bronxda.nyc.gov/html/newsroom/press-releases.shtml"
    out: list[Release] = []
    try:
        html = fetch(url)
    except Exception as e:
        print(f"[bronx] fetch failed: {e}", file=sys.stderr)
        return out
    soup = BeautifulSoup(html, "lxml")
    for a in soup.select("a[href$='.pdf']")[:MAX_RELEASES_PER_DA]:
        raw_href = a.get("href", "").strip()
        if not raw_href:
            continue
        # Some hrefs are absolute (/downloads/pdf/...), some are relative (2026/...).
        href = urljoin(url, raw_href)
        title = a.get_text(strip=True)
        # Try a few date patterns commonly seen in their filenames.
        posted = None
        for pat in (r"(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})",
                    r"pr[-_]?(20\d{2})[-_]?(\d{1,2})[-_]?(\d{1,2})"):
            m = re.search(pat, raw_href, re.IGNORECASE)
            if m:
                try:
                    posted = date(*map(int, m.groups()))
                    break
                except ValueError:
                    pass
        if href and title:
            out.append(Release(url=href, title=title, posted=posted, borough_hint="Bronx"))
    return out


SCRAPERS = [
    scrape_manhattan,
    scrape_brooklyn,
    scrape_queens,
    scrape_bronx,
    scrape_staten_island,
]


# ---------- Release body fetch ----------


def fetch_release_body(url: str) -> str:
    if url.lower().endswith(".pdf"):
        return _fetch_pdf_body(url)
    try:
        html = fetch(url)
    except Exception as e:
        print(f"[body] fetch failed for {url}: {e}", file=sys.stderr)
        return ""
    soup = BeautifulSoup(html, "lxml")
    # Drop nav/footer/script
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    main = soup.select_one("article, main, .entry-content, .post-content, #content")
    text = (main or soup).get_text("\n", strip=True)
    # Compact
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:8000]  # cap for token budget


def _fetch_pdf_body(url: str) -> str:
    """Download a PDF press release and extract its plain text via pypdf.
    Used for the Bronx DA, which publishes releases as PDFs."""
    try:
        from pypdf import PdfReader  # imported lazily to keep imports cheap
        from io import BytesIO
    except ImportError:
        print(f"[body] pypdf not installed, skipping PDF: {url}", file=sys.stderr)
        return ""
    try:
        data = fetch_bytes(url)
    except Exception as e:
        print(f"[body] PDF fetch failed for {url}: {e}", file=sys.stderr)
        return ""
    try:
        reader = PdfReader(BytesIO(data))
        pieces: list[str] = []
        # Press releases are short — first 4 pages is plenty.
        for page in reader.pages[:4]:
            try:
                pieces.append(page.extract_text() or "")
            except Exception:
                # Some PDFs have weird font tables; skip that page rather than
                # bail on the whole release.
                continue
        text = "\n".join(p for p in pieces if p.strip())
    except Exception as e:
        print(f"[body] PDF parse failed for {url}: {e}", file=sys.stderr)
        return ""
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text[:8000]


# ---------- Haiku classifier / extractor ----------


CLASSIFIER_PROMPT = """You extract felony-assault case records from New York City District Attorney press releases and named newsroom reporting for a data-journalism tracker.

A "qualifying" case means the alleged offense in the press release meets a felony-assault definition under New York Penal Law Article 120 (first, second, or aggravated-weapons third degree). This includes:
- Stabbings, shootings, severe beatings, subway pushings
- Hate-crime assaults elevated to felony
- Attacks on police/transit workers with serious physical injury
- Sex-crime cases with felony-assault counts
- Murder/attempted-murder cases where felony assault is a charged count (indictments, convictions, sentencings all count)

EXCLUDE:
- Misdemeanor-assault-only cases
- Narcotics, fraud, theft, weapon-possession-only cases (no assault count)
- Cases where the conduct happened outside NYC
- Organized-crime/racketeering press releases where no violent assault is described

Borough is one of: Manhattan, Brooklyn, Queens, Bronx, Staten Island.

Input: title, URL, posted date (if known), borough hint (the DA's borough, use unless the conduct clearly occurred elsewhere in NYC), and release body text.

Output: a JSON object exactly matching this schema, and nothing else:

{
  "qualifies": true | false,
  "reason_if_not": "short explanation if qualifies=false",
  "name": "primary defendant name(s), comma-separated; use a descriptor like '17-year-old defendant (name withheld)' if unnamed",
  "date": "YYYY-MM-DD — the date of the news event described (arrest, indictment, conviction, sentencing)",
  "borough": "Manhattan|Brooklyn|Queens|Bronx|Staten Island",
  "summary": "one to three plain-English sentences, ending in a period, describing the alleged conduct — match the voice of short wire-service style"
}

If qualifies=false, the other fields may be empty strings.

Return ONLY the JSON object, no preamble, no markdown fences.
"""


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response."""
    # Strip common fences
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find outermost braces
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def classify_and_extract(client: Anthropic, release: Release, body: str) -> dict | None:
    if not body:
        return None
    posted_str = release.posted.isoformat() if release.posted else ""
    user = (
        f"TITLE: {release.title}\n"
        f"URL: {release.url}\n"
        f"POSTED DATE: {posted_str}\n"
        f"DA BOROUGH HINT: {release.borough_hint}\n\n"
        f"BODY:\n{body}"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            system=CLASSIFIER_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        print(f"[haiku] API error for {release.url}: {e}", file=sys.stderr)
        return None
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    parsed = extract_json(text)
    if parsed is None:
        print(f"[haiku] unparseable response for {release.url}:\n{text[:300]}", file=sys.stderr)
    return parsed


# ---------- Main orchestration ----------


def load_cases() -> dict:
    with CASES_PATH.open() as f:
        return json.load(f)


def save_cases(obj: dict) -> None:
    with CASES_PATH.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — aborting", file=sys.stderr)
        return 1

    client = Anthropic(api_key=api_key)
    data = load_cases()
    cases = data.get("cases", [])
    existing_urls = {c.get("source_url") for c in cases if c.get("source_url")}

    # Determine cutoff — default to 14 days back if no prior scan
    last_scan = data.get("last_scan_date")
    cutoff = date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    if last_scan:
        try:
            cutoff = datetime.strptime(last_scan, "%Y-%m-%d").date() - timedelta(days=2)  # small safety overlap
        except ValueError:
            pass
    print(f"Scanning releases posted on or after {cutoff.isoformat()}")

    all_releases: list[Release] = []
    for scraper in SCRAPERS:
        rels = scraper()
        print(f"  {scraper.__name__}: {len(rels)} links")
        all_releases.extend(rels)

    # Filter to this year, past cutoff, and not already in our file
    candidates: list[Release] = []
    for r in all_releases:
        if r.url in existing_urls:
            continue
        if r.posted and r.posted.year != YEAR:
            continue
        if r.posted and r.posted < cutoff:
            continue
        candidates.append(r)
    print(f"Candidates after dedup/cutoff: {len(candidates)}")

    added = 0
    for rel in candidates:
        body = fetch_release_body(rel.url)
        if not body:
            continue
        extracted = classify_and_extract(client, rel, body)
        time.sleep(0.5)  # polite to the API
        if not extracted or not extracted.get("qualifies"):
            continue
        # Determine case date — fall back to posted date if Haiku didn't supply one
        case_date = extracted.get("date") or (rel.posted.isoformat() if rel.posted else None)
        if not case_date or not case_date.startswith(str(YEAR)):
            continue
        borough = extracted.get("borough") or rel.borough_hint
        if borough not in {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}:
            borough = rel.borough_hint
        entry = {
            "n": 0,  # renumbered below
            "name": extracted.get("name", "").strip() or "Defendant (name withheld)",
            "date": case_date,
            "borough": borough,
            "summary": extracted.get("summary", "").strip(),
            "source_url": rel.url,
        }
        if not entry["summary"]:
            continue
        cases.append(entry)
        existing_urls.add(rel.url)
        added += 1
        print(f"  + {entry['date']} {entry['borough']} — {entry['name']}")

    # If nothing new, don't rewrite the file — avoids noisy weekly "0 cases" commits.
    # last_scan_date advances only when the file is actually updated; the cutoff logic
    # will still look back far enough because DEFAULT_LOOKBACK_DAYS gives a floor.
    if added == 0:
        print(f"Done. No new cases. Total still {len(cases)}.")
        return 0

    # Sort chronologically, then renumber
    cases.sort(key=lambda c: (c["date"], c["name"]))
    for i, c in enumerate(cases, start=1):
        c["n"] = i

    data["cases"] = cases
    data["last_scan_date"] = date.today().isoformat()
    data["last_updated"] = date.today().isoformat()

    save_cases(data)
    print(f"Done. {added} new case(s) appended. Total now {len(cases)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
