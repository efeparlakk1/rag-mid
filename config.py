"""
config.py — Tüm RAG boru hattının merkezi ayarları.

Her aşamanın parametresi tek yerde toplanır; böylece bir değeri değiştirmek
(ör. chunk boyutu, top_k) için kod içinde dolaşmana gerek kalmaz.
"""
from __future__ import annotations
from pathlib import Path


# ── Dizinler ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"           # işlenecek PDF'ler buraya
STORAGE_DIR = BASE_DIR / "storage"     # kalıcı veriler
QDRANT_PATH = STORAGE_DIR / "qdrant"   # Qdrant embedded veritabanı dizini

DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)


# ── Cihaz (GPU/CPU) otomatik tespiti ─────────────────────────────────
def get_device() -> str:
    """CUDA varsa 'cuda', yoksa 'cpu' döndürür."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = get_device()
USE_FP16 = DEVICE == "cuda"   # fp16 yalnızca GPU'da anlamlı (hız + bellek)


# ── Embedding (BGE-M3) ───────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024              # BGE-M3 dense vektör boyutu
EMBED_MAX_LENGTH = 8192       # BGE-M3 8192 token bağlam destekler
EMBED_BATCH_SIZE = 12         # encode() iç (GPU) micro-batch boyutu
EMBED_DOC_BATCH = 32          # ingest ilerleme çubuğu için dış batch (<256 -> iç tqdm kapalı)


# ── Reranker (cross-encoder) ─────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# ── Chunking ─────────────────────────────────────────────────────────
TIKTOKEN_ENCODING = "cl100k_base"
CHUNK_TOKENS = 512           # hedef chunk boyutu (token)
CHUNK_OVERLAP = 75           # ~%15 overlap — sınır bağlamını korur
# Markdown başlıklarına göre bölme seviyeleri
MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


# ── Qdrant ───────────────────────────────────────────────────────────
COLLECTION_NAME = "docs"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ── Retrieval ────────────────────────────────────────────────────────
PREFETCH_LIMIT = 20          # dense ve sparse'ın HER BİRİNDEN çekilecek aday
RRF_LIMIT = 20               # RRF füzyonu sonrası aday havuzu boyutu
TOP_K = 5                    # reranker sonrası LLM'e gidecek nihai chunk sayısı


# ── LLM (Ollama / Qwen2.5) ───────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "qwen3.5:9b"
LLM_TEMPERATURE = 0.1        # RAG'de düşük tutulur (sadakat > yaratıcılık)
LLM_NUM_CTX = 8192           # Ollama bağlam penceresi


# ── Sohbet hafızası ──────────────────────────────────────────────────
MEMORY_WINDOW = 5            # hatırlanacak son tur (kullanıcı+asistan) sayısı
