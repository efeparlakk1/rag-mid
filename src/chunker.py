"""
Aşama 2 — Chunking: Markdown -> anlamlı parçalar (chunk)

Strateji: HİBRİT (Markdown-Header + Recursive token-bazlı)
    Text Splitting.md'deki "Senior Seviye" tavsiyesi:
    "Önce anlamsal/yapısal böl (Markdown başlıkları), eğer parça hâlâ çok
     büyükse (800+ token) onu yapısal olarak recursive böl."

İki adım:
    1) MarkdownHeaderTextSplitter: '#', '##', '###' başlıklarına göre böler.
       Her parçaya hangi başlıkların altında olduğunu metadata olarak ekler.
       -> Bu metadata citation ve "Lost in the Middle" azaltma için altın değerinde.
    2) RecursiveCharacterTextSplitter (tiktoken length): başlık bölümü hâlâ
       512 token'dan büyükse, doğal sınırlardan (paragraf->cümle->kelime)
       512 token + ~%15 overlap olacak şekilde alt parçalara böler.

Neden tiktoken? Karakter sayısı yanıltıcıdır: Türkçe/teknik içerikte 1 token
3-4 karakter olabilir. Token sayarak chunk'lar LLM context limitiyle birebir
uyumlu olur.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

import config


@dataclass
class Chunk:
    """Tek bir metin parçası ve onun zengin metadata'sı."""
    text: str
    source: str            # geldiği dosya adı
    section: str           # başlık yolu, ör. "Embedding Models > Caching"
    chunk_index: int = 0   # belge içindeki sıra (position-aware retrieval için)
    metadata: dict[str, Any] = field(default_factory=dict)


def _section_path(header_metadata: dict[str, str]) -> str:
    """{'h1': 'A', 'h2': 'B'} -> 'A > B'  (başlık yolunu okunur metne çevirir)."""
    parts = [header_metadata[key] for _, key in config.MARKDOWN_HEADERS
             if key in header_metadata]
    return " > ".join(parts) if parts else "(başlıksız)"


def chunk_markdown(markdown: str, source: str) -> list[Chunk]:
    """Markdown metnini hibrit stratejiyle Chunk listesine böler.

    Args:
        markdown: PDF'ten üretilmiş Markdown metni.
        source:   kaynak dosya adı (metadata'ya yazılır).

    Returns:
        Chunk nesnelerinin listesi.
    """
    # 1) Başlık temelli bölme — yapısal sınırlar
    #    strip_headers=False: başlık metnini içerikte bırakır; embedding'in
    #    "bu parça hangi konu hakkında?" bağlamını yakalamasına yardım eder.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=config.MARKDOWN_HEADERS,
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(markdown)

    # 2) Token-bazlı recursive bölme — büyük bölümleri kontrollü boyuta indirger
    recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.TIKTOKEN_ENCODING,
        chunk_size=config.CHUNK_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks: list[Chunk] = []
    for hdoc in header_docs:
        section = _section_path(hdoc.metadata)
        for piece in recursive_splitter.split_text(hdoc.page_content):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    section=section,
                    metadata=dict(hdoc.metadata),
                )
            )

    # Belge geneli sıra numarası ata
    for i, c in enumerate(chunks):
        c.chunk_index = i

    return chunks


if __name__ == "__main__":
    sample = (
        "# Giriş\nBu bir test belgesidir.\n\n"
        "## Bölüm 1\n" + ("Lorem ipsum dolor sit amet. " * 80) + "\n\n"
        "## Bölüm 2\nKısa bir bölüm."
    )
    out = chunk_markdown(sample, source="ornek.pdf")
    print(f"✓ {len(out)} chunk üretildi:\n")
    for c in out:
        ntok = len(c.text)
        print(f"  [#{c.chunk_index}] section='{c.section}' (~{ntok} karakter)")
