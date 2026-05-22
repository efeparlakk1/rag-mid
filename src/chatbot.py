"""
Aşama 6b — Chatbot: prompt + hafıza + query rewrite + citation

Bu modül 3 farklı .md dosyasındaki desenleri birleştirir:

  • Prompt Templates.md  -> anti-hallucination + [Doc N] citation şablonu
  • Advanced Patterns.md -> pencere hafızası (ConversationBufferWindowMemory)
                            ve "query rewriting" (takip sorusunu standalone yap)
  • Retrievers.md        -> "Lost in the Middle": en güçlü chunk'ı sıraya koy

Akış:
  kullanıcı sorusu
    │  (geçmiş varsa) query rewrite -> bağımsız soru
    ▼
  retriever.retrieve -> top-5 chunk
    │  context'i [Doc N] etiketleriyle biçimle
    ▼
  LLM -> citation'lı cevap
    │  hafızayı güncelle (son MEMORY_WINDOW tur)
    ▼
  cevap + kullanılan kaynaklar
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import config
from src.retriever import Retriever
from src.llm import LLM
from src.vector_store import RetrievedDoc

StepCallback = Callable[[str, dict], None]


# ── Prompt şablonları ────────────────────────────────────────────────
SYSTEM_TEMPLATE = """Sen yalnızca sana verilen BAĞLAM'a dayanarak Türkçe cevap veren bir RAG asistanısın.

Kurallar:
1. SADECE aşağıdaki BAĞLAM'daki bilgiyi kullan; dış bilgi ya da tahmin ekleme.
2. Her olgusal ifadenin sonunda kaynağını [Doc N] biçiminde belirt.
3. Birden fazla kaynaktan yararlandıysan her birini ayrı ayrı göster.
4. BAĞLAM soruyu yanıtlamaya yetmiyorsa AYNEN şunu yaz:
   "Sağlanan belgelerde bu sorunun cevabı bulunmuyor."
5. Net, düzenli ol; uygun yerde madde işareti kullan.

BAĞLAM:
{context}"""

REWRITE_TEMPLATE = """Aşağıda bir sohbet geçmişi ve kullanıcının son (takip) sorusu var.
Takip sorusunu, geçmişe BAKILMADAN tek başına anlaşılır, bağımsız bir soruya dönüştür.
Zamirleri (bunun, onun, o, bu) açık adlarla değiştir.
SADECE yeniden yazılmış soruyu döndür; açıklama veya ek metin yazma.

Sohbet geçmişi:
{history}

Takip sorusu: {query}
Bağımsız soru:"""


@dataclass
class ChatResult:
    answer: str
    sources: list[RetrievedDoc]
    rewritten_query: str


class Chatbot:
    def __init__(self, retriever: Retriever, llm: LLM) -> None:
        self.retriever = retriever
        self.llm = llm
        # Pencere hafızası: (kullanıcı, asistan) çiftleri
        self.history: list[tuple[str, str]] = []

    # ── Yardımcılar ──────────────────────────────────────────────────
    def _format_history(self) -> str:
        recent = self.history[-config.MEMORY_WINDOW:]
        lines = []
        for u, a in recent:
            lines.append(f"Kullanıcı: {u}")
            lines.append(f"Asistan: {a}")
        return "\n".join(lines)

    def rewrite_query(self, query: str) -> str:
        """Geçmiş varsa takip sorusunu bağımsız hale getirir (query rewriting).

        Public: hem CLI (ask) hem web arayüzü kullanır.
        """
        if not self.history:
            return query
        prompt = REWRITE_TEMPLATE.format(history=self._format_history(), query=query)
        try:
            rewritten = self.llm.generate(
                system="Sen bir soru yeniden yazma yardımcısısın.", user=prompt
            )
            return rewritten.strip() or query
        except Exception:
            return query  # rewrite başarısızsa orijinal soruya düş

    @staticmethod
    def format_context(docs: list[RetrievedDoc]) -> str:
        """Chunk'ları [Doc N] etiketleri + metadata ile LLM'e sunulacak metne çevirir."""
        blocks = []
        for i, d in enumerate(docs, start=1):
            blocks.append(
                f"[Doc {i}] (Bölüm: {d.section} | Kaynak: {d.source})\n{d.text}"
            )
        return "\n\n".join(blocks)

    def system_prompt(self, docs: list[RetrievedDoc]) -> str:
        """Anti-hallucination + citation sistem promptunu bağlamla doldurur."""
        return SYSTEM_TEMPLATE.format(context=self.format_context(docs))

    def remember(self, user: str, assistant: str) -> None:
        """Bir turu hafızaya ekler (arayüzün streaming akışından sonra çağrılır)."""
        self.history.append((user, assistant))

    # ── Ana giriş noktası (CLI / blocking) ───────────────────────────
    def ask(self, query: str, on_step: StepCallback | None = None) -> ChatResult:
        # 1) Takip sorusunu bağımsızlaştır
        standalone = self.rewrite_query(query)
        if on_step:
            on_step("rewrite", {"query": standalone, "changed": standalone != query})

        # 2) Retrieval (hibrit + rerank)
        docs = self.retriever.retrieve(standalone, on_step=on_step)

        # 3) Bağlam yoksa erken dön (hallüsinasyonu engelle)
        if not docs:
            answer = "Sağlanan belgelerde bu sorunun cevabı bulunmuyor."
            self.remember(query, answer)
            return ChatResult(answer=answer, sources=[], rewritten_query=standalone)

        # 4) Prompt'u kur ve LLM'den cevap al
        system = self.system_prompt(docs)
        answer = self.llm.generate(system=system, user=standalone)

        # 5) Hafızayı güncelle
        self.remember(query, answer)
        return ChatResult(answer=answer, sources=docs, rewritten_query=standalone)

    def reset(self) -> None:
        """Sohbet hafızasını temizler."""
        self.history.clear()
