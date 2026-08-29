"""One-off scraper for BAU's official FAQ hub pages into data/raw/web_sss/.

Each source page uses a different accordion markup, so parsing is per-source
rather than generic. Output format is plain text with `SORU:`/`CEVAP:` markers,
which src/ingestion/chunking.py's QA splitter turns into one chunk per pair.
"""

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT_DIR = Path("data/raw/web_sss")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    text = re.sub(r"^[•\-•]\s*", "", text)
    return text


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def scrape_int_faq() -> list[tuple[str, str]]:
    soup = fetch("https://int.bau.edu.tr/faq/")
    pairs = []
    for item in soup.select(".accordion-item"):
        button = item.select_one(".accordion-button")
        body = item.select_one(".accordion-body")
        if not button or not body:
            continue
        pairs.append((_clean(button.get_text()), _clean(body.get_text("\n"))))
    return pairs


def scrape_exchange_faq() -> list[tuple[str, str]]:
    soup = fetch("https://exchange.bau.edu.tr/faq/")
    pairs = []
    headers = soup.select("[data-accordion-header]")
    for header in headers:
        q_el = header.select_one("div.h6")
        content = header.find_next_sibling(attrs={"data-accordion-content": True})
        if not q_el or not content:
            continue
        pairs.append((_clean(q_el.get_text()), _clean(content.get_text("\n"))))
    return pairs


def scrape_cari_sss() -> list[tuple[str, str]]:
    soup = fetch("https://bau.edu.tr/icerik/18842-sikca-sorulan-sorular")
    pairs = []
    for item in soup.select(".faq-item"):
        q_el = item.select_one(".faq-question")
        a_el = item.select_one(".faq-answer-inner")
        if not q_el or not a_el:
            continue
        pairs.append((_clean(q_el.get_text()), _clean(a_el.get_text("\n"))))
    return pairs


def scrape_aday_faq() -> list[tuple[str, str]]:
    soup = fetch("https://aday.bau.edu.tr/sikca-sorulan-sorular/")
    pairs = []
    for card in soup.select(".sss-faq-card"):
        q_el = card.select_one(".sss-faq-title")
        a_el = card.select_one(".sss-faq-content")
        if not q_el or not a_el:
            continue
        pairs.append((_clean(q_el.get_text()), _clean(a_el.get_text("\n"))))
    return pairs


SOURCES = [
    (
        "BAU_International_FAQ.txt",
        "BAU International Student FAQ (Admission, Accommodation, Payment, Registration, "
        "Visa/Residence Permit, Transfer Student, Scholarship, Language Proficiency, Student Life)",
        scrape_int_faq,
    ),
    (
        "BAU_Exchange_Erasmus_FAQ.txt",
        "BAU Exchange & Erasmus Programs FAQ (Incoming/Outgoing Mobility, GPA Requirements, "
        "Learning Agreement, Grants)",
        scrape_exchange_faq,
    ),
    (
        "BAU_Aday_Ogrenci_SSS.txt",
        "BAU Aday Öğrenci Sıkça Sorulan Sorular (Hazırlık, Ders Kaydı, Kütüphane, Kulüpler, Burslar)",
        scrape_aday_faq,
    ),
    (
        "BAU_Cari_Hizmetler_SSS.txt",
        "BAU Öğrenci Cari Hizmetler Sıkça Sorulan Sorular (Ödemeler, Bankalar, Kayıt, Ücretler, "
        "Burslar, Yaz Okulu, İadeler, Fatura)",
        scrape_cari_sss,
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, title, scraper in SOURCES:
        pairs = scraper()
        if not pairs:
            print(f"WARNING: 0 Q&A pairs scraped for {filename}")
            continue
        lines = [title, ""]
        for question, answer in pairs:
            lines.append(f"SORU: {question}")
            lines.append(f"CEVAP: {answer}")
            lines.append("")
        (OUT_DIR / filename).write_text("\n".join(lines), encoding="utf-8")
        print(f"{filename}: {len(pairs)} Q&A pairs")


if __name__ == "__main__":
    main()
