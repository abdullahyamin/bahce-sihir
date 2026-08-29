"""One-off scraper for BAU's general informational pages (not FAQ-shaped) into
data/raw/web_genel_bilgi/: academic calendar, campus/contact directory, tuition
fees, dormitory/housing info, admin-unit directory.

Each page's real content lives in a different custom container (no shared CMS
template), so extraction is per-page via a CSS selector, then a generic
table-aware text pass (so date/label table rows stay on one line instead of
being flattened cell-by-cell).
"""

from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString

OUT_DIR = Path("data/raw/web_genel_bilgi")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

_FEEDBACK_WIDGET_STRINGS = ("Geri Bildirim Sistemi", "Bu sayfa ile ilgili değerlendirme")


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


_CANDIDATE_CONTENT_SELECTORS = [
    ".col-lg-9.mt-4",
    ".container-fluid.pt-0.pb-4.bg-white.min-vh-100",
    ".container",
]


def find_content_container(soup: BeautifulSoup):
    for selector in _CANDIDATE_CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el and len(el.get_text(strip=True)) > 50:
            return el
    return None


def extract_text(container) -> str:
    for tag in container(["script", "style"]):
        tag.decompose()
    # Collapse each table row into one "cell — cell — cell" line before the
    # final get_text pass, so paired data (date — event) survives as one unit.
    for tr in container.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        tr.replace_with(NavigableString(" — ".join(cells) + "\n" if cells else ""))

    text = container.get_text("\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines = [l for l in lines if not any(w in l for w in _FEEDBACK_WIDGET_STRINGS)]
    # de-duplicate immediate consecutive repeats (nested tags often echo the same line)
    deduped = []
    for l in lines:
        if not deduped or deduped[-1] != l:
            deduped.append(l)
    return "\n".join(deduped)


SOURCES = [
    (
        "BAU_Akademik_Takvim_Onlisans_Lisans.txt",
        "BAU 2026-2027 Akademik Yılı Önlisans ve Lisans Akademik Takvimi",
        "https://bau.edu.tr/icerik/18785-2026-2027-akademik-yili-onlisans-ve-lisans-egitim-ogretimi-akademik-takvimi",
        ".academic-calendar-wrapper",
    ),
    (
        "BAU_Akademik_Takvim_Lisansustu.txt",
        "BAU 2026-2027 Akademik Yılı Lisansüstü Akademik Takvimi",
        "https://bau.edu.tr/icerik/18787-2026-2027-akademik-yili-lisansustu-egitim-ve-ogretimi-akademik-takvimi",
        ".academic-calendar-wrapper",
    ),
    (
        "BAU_Kampusler_Iletisim_Ulasim.txt",
        "BAU Kampüsler, Adresler, İletişim ve Ulaşım Bilgileri",
        "https://bau.edu.tr/iletisim-ulasim",
        "#nav-tabContent",
    ),
    (
        "BAU_Onlisans_Lisans_Ucretleri_2026_2027.txt",
        "BAU 2026-2027 Önlisans ve Lisans Ücretleri",
        "https://aday.bau.edu.tr/onlisans-ve-lisans-ucretler-2026-2027/",
        ".fees-container",
    ),
    (
        "BAU_Yurt_Imkanlari.txt",
        "BAU Yurt ve Konaklama İmkanları (BAU Ortaköy Aparts ve Anlaşmalı Yurtlar)",
        "https://aday.bau.edu.tr/yurt-imkanlari/",
        ".bau-yurt-container",
    ),
    (
        "BAU_Idari_Birimler.txt",
        "BAU İdari Birimler Rehberi (Daire Başkanlıkları, Direktörlükler, Koordinatörlükler)",
        "https://bau.edu.tr/icerik/2591-idari-birimler",
        ".container",
    ),
    (
        "BAU_Insan_Kaynaklari_Iletisim.txt",
        "BAU İnsan Kaynakları Daire Başkanlığı İletişim",
        "https://bau.edu.tr/icerik/13786-insan-kaynaklari-iletisim",
        ".col-lg-9.mt-4",
    ),
    (
        "BAU_Kurumsal_Iletisim.txt",
        "BAU Kurumsal ve Stratejik İletişim Daire Başkanlığı",
        "https://bau.edu.tr/icerik/18779-kurumsal-iletisim",
        ".col-lg-9.mt-4",
    ),
    (
        "BAU_Ogrenci_Isleri_Formlar.txt",
        "BAU Öğrenci İşleri Daire Başkanlığı - Formlar",
        "https://bau.edu.tr/icerik/2636-ogrenci-isleri-formlar",
        ".col-lg-9.mt-4",
    ),
    (
        "BAU_BIDB_Iletisim.txt",
        "BAU Bilgi İşlem Daire Başkanlığı İletişim",
        "https://bidb.bau.edu.tr/contact-us/",
        ".elementor-container",
    ),
]


# AKTS "Bilgi Paketi" — one consistent URL pattern per page id, content always in #form.
# menu_id doesn't actually need to match id (verified: the server keys content off id
# alone) but is passed anyway to mirror the site's own links.
AKTS_PAGES = [
    (6, "1_1", "İsim ve Adres Bilgileri"),
    (8, "1_3", "Üniversite Yönetimi"),
    (1, "1_4", "Hakkımızda"),
    (12, "1_6", "Kabul Koşulları"),
    (9, "1_7", "Önceki Öğrenmenin Tanınması"),
    (13, "1_8", "Ders Kayıt Prosedürü"),
    (15, "1_9", "AKTS Dağılımı"),
    (10, "1_10", "Yurt Dışında Eğitimin Tanınması"),
    (17, "1_11", "Akademik Danışmanlık"),
    (16, "1_12", "Ulusal Kredi ve AKTS Değerleri"),
    (18, "1_13", "Uluslararasılaşma"),
    (38, "1_14", "Hayat Boyu Öğrenme"),
    (39, "1_15", "Araştırma ve Uygulama Merkezleri"),
    (40, "1_16", "BAU'da Kültür ve Sanat"),
    (41, "1_17", "BAU Yayınlar"),
    (36, "1_18", "Kurum AKTS Koordinatörü"),
    (37, "1_19", "Bölüm AKTS Koordinatörleri"),
    (19, "2_24", "İstanbul Hakkında"),
    (20, "2_25", "Barınma"),
    (21, "2_26", "Yemek"),
    (22, "2_27", "Sağlık Hizmetleri"),
    (23, "2_28", "Engelli Öğrenci Hizmetleri"),
    (24, "2_29", "Kariyer Yönlendirme Merkezi"),
    (25, "2_30", "Sigorta"),
    (26, "2_31", "Öğrenciler için Finansal Destek"),
    (27, "2_32", "Öğrenci İşleri"),
    (28, "2_33", "Öğrenim Hizmetleri"),
    (29, "2_34", "Uluslararası Programlar"),
    (30, "2_35", "Değişim Öğrencileri için Pratik Bilgiler"),
    (31, "2_36", "Dil Dersleri"),
    (32, "2_37", "Stajlar"),
    (33, "2_38", "Spor ve Sosyal Yaşam"),
    (34, "2_39", "Öğrenci Kulüpleri"),
    (35, "2_40", "Faydalı Telefon Numaraları"),
]

# exchange.bau.edu.tr program-specific pages — consistent .wrap.formatted content container.
EXCHANGE_PAGES = [
    ("erasmus/incoming-students/", "Erasmus+ Incoming Students"),
    ("erasmus/outgoing-students/", "Erasmus+ Outgoing Study"),
    ("erasmus/outgoing-students/partner-universities-programs/", "Erasmus+ Partner Universities & Programs"),
    ("erasmus-traineeship/", "Erasmus+ Outgoing Traineeship"),
    ("erasmus-internship4all-consortium/", "Erasmus+ Internship4All Consortium"),
    ("ka171/", "Erasmus+ KA171"),
    ("general-information-for-erasmus-ka171/", "General Information for Erasmus+ KA171"),
    ("anlasmalar/", "Erasmus+ KA171 Partner Universities"),
    ("global-programs/", "Global Programs"),
    ("besl/", "BESL (Study Abroad English Program)"),
    ("study-abroad-washington-d-c/", "Study Abroad Washington D.C."),
    ("berlin-exchange/", "Berlin Exchange"),
    ("dual-up/", "Dual Up"),
    ("mou-exchange/", "MoU Exchange"),
    ("orhun-exchange/", "Orhun Exchange"),
    ("faculty-programs/", "Faculty Programs"),
    ("summer-winter-schools/", "Summer/Winter Schools"),
    ("mevlana-degisim-programi/", "Mevlana Exchange Program"),
]


# Per-department internship (staj) pages — same bau.edu.tr CMS template as the general
# pages, content lives in a sidebar-content-column div (.col-lg-9.mt-4).
STAJ_PAGES = [
    ("icerik/3429-endustri-muhendisligi-staj", "BAU_Staj_Endustri_Muhendisligi.txt"),
    ("icerik/14602-staj-isletme-muhendisligi", "BAU_Staj_Isletme_Muhendisligi.txt"),
    ("icerik/12459-insaat-muhendisligi-staj", "BAU_Staj_Insaat_Muhendisligi.txt"),
    ("icerik/15858-staj-yazilim-muhendisligi", "BAU_Staj_Yazilim_Muhendisligi.txt"),
    ("icerik/13533-elektrik-elektronik-muhendisligi-staj", "BAU_Staj_Elektrik_Elektronik_Muhendisligi.txt"),
    ("haber/13424-bilgisayar-muhendisligi-staj", "BAU_Staj_Bilgisayar_Muhendisligi.txt"),
    ("icerik/12446-staj-bilgileri-ve-evraklari", "BAU_Staj_Bilgileri_ve_Evraklari.txt"),
    ("icerik/13241-zorunlu-staj", "BAU_Staj_Zorunlu_Staj_Genel.txt"),
    ("icerik/15897-staj", "BAU_Staj_Genel_Hub.txt"),
    ("icerik/14647-internship", "BAU_Internship_General_EN.txt"),
]


def scrape_staj_pages() -> None:
    for path, filename in STAJ_PAGES:
        url = f"https://bau.edu.tr/{path}"
        try:
            soup = fetch(url)
        except requests.RequestException as e:
            print(f"WARNING: fetch failed for {filename}: {e}")
            continue
        container = find_content_container(soup)
        if not container:
            print(f"WARNING: no content container found for {filename}")
            continue
        body = extract_text(container)
        if len(body) < 100:
            print(f"WARNING: suspiciously short content ({len(body)} chars) for {filename}")
        (OUT_DIR / filename).write_text(body, encoding="utf-8")
        print(f"{filename}: {len(body)} chars")


def scrape_static_sources() -> None:
    for filename, title, url, selector in SOURCES:
        soup = fetch(url)
        container = soup.select_one(selector)
        if not container:
            print(f"WARNING: selector {selector!r} not found for {filename}")
            continue
        body = extract_text(container)
        if len(body) < 50:
            print(f"WARNING: suspiciously short content ({len(body)} chars) for {filename}")
        (OUT_DIR / filename).write_text(f"{title}\n\n{body}", encoding="utf-8")
        print(f"{filename}: {len(body)} chars")


def scrape_akts_pages() -> None:
    for page_id, menu_id, title in AKTS_PAGES:
        url = f"https://akts.bau.edu.tr/bilgipaketi/index/icerik/id/{page_id}/menu_id/{menu_id}/ln/tr"
        filename = f"BAU_AKTS_{title.replace(' ', '_').replace('/', '-')}.txt"
        try:
            soup = fetch(url)
        except requests.RequestException as e:
            print(f"WARNING: fetch failed for {filename}: {e}")
            continue
        container = soup.select_one("#form")
        if not container:
            print(f"WARNING: #form not found for {filename}")
            continue
        body = extract_text(container)
        if len(body) < 30:
            print(f"WARNING: suspiciously short content ({len(body)} chars) for {filename}")
        (OUT_DIR / filename).write_text(body, encoding="utf-8")
        print(f"{filename}: {len(body)} chars")


def scrape_exchange_pages() -> None:
    for path, title in EXCHANGE_PAGES:
        url = f"https://exchange.bau.edu.tr/{path}"
        filename = f"BAU_Exchange_{title.replace(' ', '_').replace('/', '-').replace('.', '')}.txt"
        try:
            soup = fetch(url)
        except requests.RequestException as e:
            print(f"WARNING: fetch failed for {filename}: {e}")
            continue
        container = soup.select_one(".wrap.formatted")
        if not container:
            print(f"WARNING: .wrap.formatted not found for {filename}")
            continue
        body = extract_text(container)
        if len(body) < 30:
            print(f"WARNING: suspiciously short content ({len(body)} chars) for {filename}")
        (OUT_DIR / filename).write_text(f"{title}\n\n{body}", encoding="utf-8")
        print(f"{filename}: {len(body)} chars")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scrape_static_sources()
    scrape_akts_pages()
    scrape_exchange_pages()
    scrape_staj_pages()


if __name__ == "__main__":
    main()
