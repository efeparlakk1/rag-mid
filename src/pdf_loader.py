"""
Aşama 1 — Document Loading: Kaynak dosya -> Markdown

Neden Markdown?
    Document Loading.md'deki uzman tavsiyesi: "Dokümanları her zaman Markdown'a
    çevirmeye çalış. LLM'ler başlık hiyerarşisini (H1, H2), tabloları ve bold
    yapıları Markdown'da çok daha iyi anlar."

Bu modül artık İKİ formatı destekler ve uzantıya göre yönlendirir (dispatch):
    .pdf            -> PyMuPDF4LLM ile Markdown'a çevrilir
    .md / .markdown -> zaten Markdown; doğrudan okunur (encoding tespitiyle)

PyMuPDF4LLM, PyMuPDF'in (fitz) üzerine kurulu bir katmandır ve PDF'in görsel
yapısını (başlık font boyutları, listeler, tablolar) analiz ederek doğrudan
Markdown üretir. Markdown başlık hiyerarşisini koruması, bir sonraki aşamada
(chunking) "Markdown-header" bölmesini mümkün kılan kritik özelliktir. .md
dosyaları zaten bu yapıya sahip olduğundan ek bir parse adımına gerek kalmaz.
"""
from __future__ import annotations
from pathlib import Path

import pymupdf4llm

# Sistemin işleyebildiği kaynak uzantıları
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


def pdf_to_markdown(pdf_path: str | Path) -> str:
    """Bir PDF dosyasını tek bir Markdown metnine çevirir.

    Args:
        pdf_path: PDF dosyasının yolu.

    Returns:
        Başlık hiyerarşisi korunmuş Markdown metni.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF bulunamadı: {pdf_path}")

    # to_markdown: sayfaları sırayla işler, başlıkları # / ## olarak işaretler.
    markdown = pymupdf4llm.to_markdown(str(pdf_path))

    if not markdown or not markdown.strip():
        # Document Loading.md: "Şifreli PDF'ler sessizce boş metin döndürebilir."
        # Burada erken yakalıyoruz ki boş içerik index'e sızmasın.
        raise ValueError(
            f"'{pdf_path.name}' boş metin üretti. PDF taranmış (OCR gerekli) "
            "veya şifreli olabilir."
        )

    return markdown


def _read_text_file(path: Path) -> str:
    """Bir metin dosyasını encoding'i tahmin ederek okur.

    Document Loading.md: "Encoding'i asla varsayma." Önce UTF-8 (BOM dahil),
    sonra charset-normalizer ile otomatik tespit, en son latin-1 denenir.
    """
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except Exception:
        pass
    return raw.decode("latin-1", errors="replace")


def load_as_markdown(path: str | Path) -> str:
    """Kaynak dosyayı uzantısına göre Markdown metnine çevirir (dispatcher).

    .pdf            -> PyMuPDF4LLM ile çevrilir
    .md / .markdown -> doğrudan okunur

    Args:
        path: kaynak dosyanın yolu.

    Returns:
        Markdown metni (her iki format için tek tip çıktı).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return pdf_to_markdown(path)

    if ext in (".md", ".markdown"):
        text = _read_text_file(path)
        if not text.strip():
            raise ValueError(f"'{path.name}' boş bir Markdown dosyası.")
        return text

    raise ValueError(
        f"Desteklenmeyen format: '{ext}'. "
        f"Desteklenenler: {sorted(SUPPORTED_EXTENSIONS)}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python -m src.pdf_loader <pdf_veya_md_yolu>")
        sys.exit(1)

    md = load_as_markdown(sys.argv[1])
    print(f"✓ {len(md)} karakter Markdown üretildi.\n")
    print(md[:1000])
