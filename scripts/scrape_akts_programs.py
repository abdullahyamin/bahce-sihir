"""Scrapes BAU's AKTS course-catalog program list (akts.bau.edu.tr) — one short
goal/description paragraph per degree program, across all 4 degree levels.
Grouped into 4 SORU:/CEVAP:-formatted files (one per level) in data/raw/web_sss/,
so each program becomes its own chunk via the existing QA chunker.
"""

import time

import requests
from bs4 import BeautifulSoup

OUT_DIR = __import__("pathlib").Path("data/raw/web_sss")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
BASE = "https://akts.bau.edu.tr"

LEVELS = [
    ("OL", "3_20", "Önlisans", "BAU_AKTS_Onlisans_Programlari.txt"),
    ("L", "3_21", "Lisans", "BAU_AKTS_Lisans_Programlari.txt"),
    ("D", "3_22", "Doktora", "BAU_AKTS_Doktora_Programlari.txt"),
    ("YL", "3_23", "Yüksek Lisans", "BAU_AKTS_Yuksek_Lisans_Programlari.txt"),
]


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def list_programs(level_code: str, menu_id: str) -> list[tuple[str, str, str]]:
    """Returns list of (faculty_name, program_name, program_href)."""
    soup = fetch(f"{BASE}/bilgipaketi/index/akademik/tip/{level_code}/menu_id/{menu_id}/ln/tr")
    form = soup.select_one("#form")
    if not form:
        return []
    out = []
    current_faculty = ""
    for a in form.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if "fakultetanimi" in href:
            current_faculty = text
        elif "programtanimi" in href:
            out.append((current_faculty, text, href))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for level_code, menu_id, level_label, filename in LEVELS:
        programs = list_programs(level_code, menu_id)
        print(f"{level_label}: {len(programs)} programs")

        lines = [f"BAU AKTS {level_label} Programları (Program Hedefleri)", ""]
        for faculty, program_name, href in programs:
            try:
                soup = fetch(f"{BASE}{href}")
            except requests.RequestException as e:
                print(f"  WARNING: fetch failed for {program_name}: {e}")
                continue
            form = soup.select_one("#form")
            desc = form.get_text(" ", strip=True) if form else ""
            if len(desc) < 20:
                print(f"  WARNING: empty description for {program_name}")
                time.sleep(0.3)
                continue
            faculty_part = f" ({faculty})" if faculty else ""
            lines.append(f"SORU: {program_name}{faculty_part} {level_label} programının hedefi/amacı nedir?")
            lines.append(f"CEVAP: {desc}")
            lines.append("")
            time.sleep(0.3)

        (OUT_DIR / filename).write_text("\n".join(lines), encoding="utf-8")
        print(f"{filename}: saved {len(programs)} program entries")


if __name__ == "__main__":
    main()
