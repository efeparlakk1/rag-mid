"""
Aşama 6a — LLM istemcisi: Ollama / Qwen2.5 (local)

Local LLM seçildi: veri dışarı çıkmaz, API ücreti yok. Ollama, modeli yerel
sunucuda (http://localhost:11434) servis eder; biz HTTP üzerinden konuşuruz.

Structured Data.md / RAG Core.md: RAG'de temperature DÜŞÜK tutulur — model
"yaratıcı" davranıp context'ten uzaklaşmasın, sağlanan bilgiye sadık kalsın.
"""
from __future__ import annotations

import ollama

import config


class LLM:
    def __init__(self) -> None:
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.model = config.LLM_MODEL

    def is_available(self) -> bool:
        """Ollama sunucusu ayakta ve model indirilmiş mi?"""
        try:
            models = self.client.list().get("models", [])
            names = {m.get("model", "") for m in models}
            # "qwen2.5:7b" veya "qwen2.5:7b-instruct" gibi varyantları tolere et
            return any(n.startswith(self.model.split(":")[0]) for n in names)
        except Exception:
            return False

    def _messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _options(self) -> dict:
        return {
            "temperature": config.LLM_TEMPERATURE,
            "num_ctx": config.LLM_NUM_CTX,
        }

    def generate(self, system: str, user: str) -> str:
        """Sistem + kullanıcı mesajıyla tek seferlik yanıt üretir (blocking)."""
        resp = self.client.chat(
            model=self.model,
            messages=self._messages(system, user),
            options=self._options(),
        )
        return resp["message"]["content"].strip()

    def generate_stream(self, system: str, user: str):
        """Yanıtı token-token üreten generator (arayüzde canlı akış için)."""
        stream = self.client.chat(
            model=self.model,
            messages=self._messages(system, user),
            options=self._options(),
            stream=True,
        )
        for part in stream:
            piece = part.get("message", {}).get("content", "")
            if piece:
                yield piece
