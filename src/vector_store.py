"""
Aşama 4 — Vektör Deposu: Qdrant (embedded)

Vector Stores.md: Qdrant disk-persistent, Rust tabanlı, NATIVE SPARSE vektör
destekler ve dense+sparse'ı tek sorguda RRF ile birleştirebilir. BGE-M3 ile
birlikte "gerçek hibrit arama"yı tek bir API çağrısında kurmamızı sağlar.

Embedded mod: ayrı sunucu/Docker gerekmez. `QdrantClient(path=...)` ile veriler
diske yazılır (mmap). Orta seviye + tek makine için ideal.

Koleksiyon iki isimli vektör tutar:
    - "dense"  : 1024-dim, COSINE mesafe (semantik)
    - "sparse" : seyrek vektör (öğrenilmiş keyword)
"""
from __future__ import annotations
from dataclasses import dataclass
import uuid

from qdrant_client import QdrantClient, models
from tqdm import tqdm

import config
from src.embedder import SparseVector


@dataclass
class RetrievedDoc:
    """Aramadan dönen tek bir sonuç (içerik + metadata + skor)."""
    text: str
    source: str
    section: str
    chunk_index: int
    score: float


class LockedStoreError(RuntimeError):
    """Qdrant embedded deposu başka bir süreç tarafından açıkken fırlatılır."""


_LOCK_HELP = (
    "Qdrant deposu (storage/qdrant) şu anda BAŞKA bir süreçte açık.\n"
    "  Embedded mod aynı anda yalnızca tek süreç destekler.\n"
    "  Çözüm:\n"
    "   • Açık olan chat.py / 'streamlit run app.py' / ingest.py'yi kapat.\n"
    "   • Takılı kaldıysa:  pkill -f streamlit   (ve/veya ilgili python sürecini)\n"
    "   • Sonra komutu tekrar çalıştır."
)


class VectorStore:
    def __init__(self) -> None:
        # Embedded mod: path verince yerel diske kalıcı yazar.
        try:
            self.client = QdrantClient(path=str(config.QDRANT_PATH))
        except RuntimeError as e:
            if "already accessed" in str(e).lower():
                raise LockedStoreError(_LOCK_HELP) from e
            raise

    # ── Koleksiyon yönetimi ──────────────────────────────────────────
    def recreate_collection(self) -> None:
        """Koleksiyonu sıfırdan kurar (varsa siler). Tam ingest öncesi çağrılır."""
        if self.client.collection_exists(config.COLLECTION_NAME):
            self.client.delete_collection(config.COLLECTION_NAME)

        self.client.create_collection(
            collection_name=config.COLLECTION_NAME,
            # Dense vektör: COSINE — Embedding Models.md'nin RAG default tavsiyesi
            vectors_config={
                config.DENSE_VECTOR_NAME: models.VectorParams(
                    size=config.EMBED_DIM,
                    distance=models.Distance.COSINE,
                ),
            },
            # Sparse vektör: BGE-M3 ağırlıkları zaten öğrenilmiş, IDF modifier yok.
            sparse_vectors_config={
                config.SPARSE_VECTOR_NAME: models.SparseVectorParams(),
            },
        )

    # ── Yazma ────────────────────────────────────────────────────────
    def upsert(
        self,
        texts: list[str],
        sources: list[str],
        sections: list[str],
        chunk_indices: list[int],
        dense_vecs,                      # np.ndarray (N, 1024)
        sparse_vecs: list[SparseVector],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> int:
        """Chunk'ları dense+sparse vektörleriyle Qdrant'a yazar."""
        points: list[models.PointStruct] = []
        for text, src, sec, ci, dvec, svec in zip(
            texts, sources, sections, chunk_indices, dense_vecs, sparse_vecs
        ):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        config.DENSE_VECTOR_NAME: dvec.tolist(),
                        config.SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=svec.indices, values=svec.values
                        ),
                    },
                    payload={
                        "text": text,
                        "source": src,
                        "section": sec,
                        "chunk_index": ci,
                    },
                )
            )

        # Büyük yüklemelerde tek tek değil toplu (batch) yaz — Vector Stores.md:
        # "Batched upsert performansı dramatik artırır."
        bar = tqdm(
            total=len(points),
            desc="📦 Qdrant'a yazılıyor",
            unit="vec",
            colour="#b14dff",          # neon mor
            disable=not show_progress,
            ncols=88,
        )
        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            self.client.upsert(collection_name=config.COLLECTION_NAME, points=chunk)
            bar.update(len(chunk))
        bar.close()
        return len(points)

    # ── Hibrit arama (dense + sparse, RRF füzyon) ────────────────────
    def hybrid_search(
        self,
        dense_query,                     # np.ndarray (1024,)
        sparse_query: SparseVector,
        limit: int | None = None,
    ) -> list[RetrievedDoc]:
        """Dense ve sparse adaylarını çekip Qdrant'ın native RRF'i ile birleştirir.

        RRF (Reciprocal Rank Fusion): her listedeki SIRAYA bakar, skor ölçeği
        farkını umursamaz -> dense (0-1 cosine) ile sparse (büyük sayılar) sorunsuz
        birleşir. Embedding Models.md: "normalize gerektirmez, basit ve robust."
        """
        limit = limit or config.RRF_LIMIT
        result = self.client.query_points(
            collection_name=config.COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_query.tolist(),
                    using=config.DENSE_VECTOR_NAME,
                    limit=config.PREFETCH_LIMIT,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices, values=sparse_query.values
                    ),
                    using=config.SPARSE_VECTOR_NAME,
                    limit=config.PREFETCH_LIMIT,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return [
            RetrievedDoc(
                text=p.payload["text"],
                source=p.payload["source"],
                section=p.payload["section"],
                chunk_index=p.payload["chunk_index"],
                score=p.score,
            )
            for p in result.points
        ]

    def count(self) -> int:
        return self.client.count(config.COLLECTION_NAME).count

    def close(self) -> None:
        """Embedded modda dosya kilidini düzgün bırakır."""
        try:
            self.client.close()
        except Exception:
            pass
