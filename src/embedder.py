"""
Aşama 3 — Embedding: metin -> vektör (BGE-M3)

Embedding Models.md'ye göre BGE-M3'ün gücü: TEK MODEL, TEK EMBEDDING UZAYI
içinde hem dense hem sparse üretmesi. Klasik hibrit mimari 3 ayrı model
(dense + BM25 + reranker) gerektirir; BGE-M3 dense ve sparse'ı tek encode
çağrısında verir -> normalizasyon/ölçek sorunu olmaz.

    - dense_vecs     : (N, 1024) — semantik anlam (cosine ile karşılaştırılır)
    - lexical_weights: {token_id: ağırlık} — öğrenilmiş sparse (keyword + term
                       expansion). BM25'ten farkı: "car" için "vehicle" gibi
                       ilgili terimleri de aktive eder.

Bu ikisini Qdrant'a yazıp native RRF ile birleştireceğiz (gerçek hibrit arama).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import config


@dataclass
class SparseVector:
    """Qdrant'ın beklediği seyrek vektör formatı: indeks + değer listeleri."""
    indices: list[int]
    values: list[float]


class Embedder:
    """BGE-M3 sarmalayıcısı. Model bir kez yüklenir, tekrar tekrar kullanılır."""

    def __init__(self) -> None:
        # İçe aktarmayı __init__ içinde yapıyoruz ki modül import edilince
        # (ör. testlerde) ağır torch yüklemesi tetiklenmesin.
        from FlagEmbedding import BGEM3FlagModel

        print(f"[embedder] BGE-M3 yükleniyor (device={config.DEVICE}, fp16={config.USE_FP16})...")
        self.model = BGEM3FlagModel(
            config.EMBED_MODEL,
            use_fp16=config.USE_FP16,
            devices=config.DEVICE,
        )

    # ── Düşük seviye encode ──────────────────────────────────────────
    def _encode(self, texts: list[str]) -> dict:
        return self.model.encode(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            max_length=config.EMBED_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # orta seviye için ColBERT'i kullanmıyoruz
        )

    @staticmethod
    def _to_sparse(lexical_weights: dict) -> SparseVector:
        """BGE-M3'ün {token_id_str: weight} çıktısını Qdrant formatına çevirir."""
        indices = [int(tok) for tok in lexical_weights.keys()]
        values = [float(w) for w in lexical_weights.values()]
        return SparseVector(indices=indices, values=values)

    # ── Belgeler için (toplu) ────────────────────────────────────────
    def embed_documents(
        self, texts: list[str], show_progress: bool = False
    ) -> tuple[np.ndarray, list[SparseVector]]:
        """Birden çok metni dense matris + sparse vektör listesine çevirir.

        İlerleme çubuğu için metinler EMBED_DOC_BATCH'lik gruplar halinde
        encode edilir (her grup <256 olduğundan FlagEmbedding'in iç bar'ı kapalı
        kalır; bizim renkli tqdm'imiz tek gösterge olur).
        """
        bs = config.EMBED_DOC_BATCH
        dense_parts: list[np.ndarray] = []
        sparse_all: list[SparseVector] = []

        bar = tqdm(
            total=len(texts),
            desc="🧬 Embedding (dense+sparse)",
            unit="chunk",
            colour="#00e5ff",          # neon camgöbeği
            disable=not show_progress,
            ncols=88,
        )
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            out = self._encode(batch)
            dense_parts.append(out["dense_vecs"])
            sparse_all.extend(self._to_sparse(lw) for lw in out["lexical_weights"])
            bar.update(len(batch))
        bar.close()

        dense = np.vstack(dense_parts) if dense_parts else np.empty((0, config.EMBED_DIM))
        return dense, sparse_all

    # ── Sorgu için (tek) ─────────────────────────────────────────────
    def embed_query(self, text: str) -> tuple[np.ndarray, SparseVector]:
        """Tek bir sorguyu dense vektör + sparse vektöre çevirir."""
        dense, sparse = self.embed_documents([text])
        return dense[0], sparse[0]


if __name__ == "__main__":
    emb = Embedder()
    d, s = emb.embed_query("RAG sisteminde hibrit arama nedir?")
    print(f"✓ dense boyut: {d.shape}")
    print(f"✓ sparse aktif terim sayısı: {len(s.indices)}")
