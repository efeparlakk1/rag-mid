"""
ingest.py — INDEXLEME boru hattı (offline aşama)

Bir veya birden çok kaynağı (PDF ve/veya Markdown) aramaya hazır hale getirir:
    kaynak -> Markdown -> Chunk -> Embed (dense+sparse) -> Qdrant'a yaz

Artık hem .pdf hem .md/.markdown destekler. Birden fazla dosya veya bir klasör
verebilirsin; klasör özyinelemeli (recursive) taranır.

Kullanım:
    python ingest.py data/dosya.pdf
    python ingest.py data/notlar.md
    python ingest.py data/                       # data/ içindeki tüm pdf+md
    python ingest.py data/a.pdf data/b.md        # birden çok dosya
    python ingest.py data/yeni.md --keep         # koleksiyonu silmeden ekle
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from tqdm import tqdm

from src.pdf_loader import load_as_markdown, SUPPORTED_EXTENSIONS
from src.chunker import chunk_markdown
from src.embedder import Embedder
from src.vector_store import VectorStore, LockedStoreError


def discover_files(paths: list[str]) -> list[Path]:
    """Verilen yolları (dosya/klasör) desteklenen kaynak dosyalarına genişletir.

    Document Loading.md: "Klasör yapısı doğal metadatadır; uzantıya göre doğru
    loader'a dispatch et." Burada uzantıya göre filtreliyoruz.
    """
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(f)
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
            else:
                print(f"  ! atlandı (desteklenmeyen format): {p.name}")
        else:
            print(f"  ! atlandı (bulunamadı): {p}")
    return files


def ingest(paths: list[str], keep: bool = False) -> None:
    t0 = time.time()

    files = discover_files(paths)
    if not files:
        print("Desteklenen kaynak bulunamadı (.pdf / .md / .markdown).")
        return
    print(f"İşlenecek {len(files)} kaynak bulundu:")
    for f in files:
        print(f"  • {f.name}  ({f.suffix.lower()})")
    print()

    # Modelleri ve veritabanını BİR KEZ yükle (tüm dosyalar için tekrar kullan)
    embedder = Embedder()
    try:
        store = VectorStore()
    except LockedStoreError as e:
        print(f"\n! {e}")
        return
    if not keep:
        store.recreate_collection()  # tam yeniden index: koleksiyonu sıfırla

    grand_total = 0
    # Dosyalar arası genel ilerleme (yeşil) — alt bar'ların üstünde durur
    for i, f in enumerate(files, start=1):
        tqdm.write(f"\n\033[1;92m[{i}/{len(files)}] {f.name}\033[0m")
        try:
            # 1) Kaynak -> Markdown (uzantıya göre otomatik)
            markdown = load_as_markdown(f)

            # 2) Markdown -> Chunk'lar (+ başlık metadata)
            chunks = chunk_markdown(markdown, source=f.name)
            if not chunks:
                tqdm.write("      ! chunk üretilmedi, atlanıyor.")
                continue
            tqdm.write(f"      • {len(chunks)} chunk üretildi")

            # 3) Embedding (dense + sparse) — renkli ilerleme çubuğu
            dense, sparse = embedder.embed_documents(
                [c.text for c in chunks], show_progress=True
            )

            # 4) Qdrant'a yaz (her dosya koleksiyona EKLENİR) — renkli çubuk
            n = store.upsert(
                texts=[c.text for c in chunks],
                sources=[c.source for c in chunks],
                sections=[c.section for c in chunks],
                chunk_indices=[c.chunk_index for c in chunks],
                dense_vecs=dense,
                sparse_vecs=sparse,
                show_progress=True,
            )
            grand_total += n
        except Exception as e:
            # Document Loading.md: "Tek bozuk doküman tüm pipeline'ı durdurmamalı."
            tqdm.write(f"      ! HATA, bu dosya atlandı: {e}")
            continue

    total = store.count()
    store.close()
    dt = time.time() - t0
    print(f"\n✓ Tamamlandı: {grand_total} chunk yazıldı "
          f"(koleksiyon toplamı: {total}). Süre: {dt:.1f}s")
    print(f"  Şimdi sohbet için:  python chat.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF ve/veya Markdown kaynaklarını RAG için indexle"
    )
    parser.add_argument(
        "paths", nargs="+",
        help="Dosya(lar) veya klasör(ler): .pdf / .md / .markdown (ör. data/)",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Mevcut koleksiyonu silme, üzerine ekle",
    )
    args = parser.parse_args()
    ingest(args.paths, keep=args.keep)
