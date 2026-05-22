"""
Aşama 5a — Reranking: cross-encoder ile yeniden sıralama

Embedding Models.md / Retrievers.md: İki aşamalı retrieval production standardı.
    1. aşama (bi-encoder + hibrit): geniş aday seti getir (recall odaklı, hızlı)
    2. aşama (cross-encoder): adayları yeniden sırala (precision odaklı, yavaş)

Fark: bi-encoder query ve dokümanı AYRI encode eder (etkileşimi göremez).
Cross-encoder query+dokümanı BİRLİKTE encode eder, aralarındaki ilişkiyi
doğrudan görür -> çok daha doğru. Pahalı olduğu için yalnızca 1. aşamanın
daralttığı ~20 aday üzerinde çalıştırılır.

Model: bge-reranker-v2-m3 — çok dilli (Türkçe dahil), BGE-M3 ailesiyle uyumlu.
Implementasyon: sentence-transformers `CrossEncoder` (güncel transformers ile
uyumlu, hızlı tokenizer kullanır).
"""
from __future__ import annotations

import numpy as np

import config
from src.vector_store import RetrievedDoc


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Cross-encoder logit'lerini okunur 0-1 skoruna çevirir."""
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        print(f"[reranker] bge-reranker-v2-m3 yükleniyor (device={config.DEVICE})...")
        self.model = CrossEncoder(
            config.RERANKER_MODEL,
            max_length=512,
            device=config.DEVICE,
        )

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int | None = None,
    ) -> list[RetrievedDoc]:
        """Adayları (query, doküman) çiftleri olarak puanlayıp en iyi top_k'yı döndürür."""
        top_k = top_k or config.TOP_K
        if not docs:
            return []

        pairs = [(query, d.text) for d in docs]
        # activation_fn=None -> ham logit döndür; sigmoid'i biz uygularız.
        logits = self.model.predict(pairs, activation_fn=None, convert_to_numpy=True)
        scores = _sigmoid(np.asarray(logits, dtype=float)).tolist()

        # Cross-encoder skorunu doc'a yaz ve buna göre sırala
        for d, s in zip(docs, scores):
            d.score = float(s)

        ranked = sorted(docs, key=lambda d: d.score, reverse=True)
        return ranked[:top_k]
