"""Central configuration — loaded from .env via python-dotenv."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Backend selection ─────────────────────────────────────────────────────────
EMBED_BACKEND    = os.getenv("EMBED_BACKEND",    "ollama")
CHAT_BACKEND     = os.getenv("CHAT_BACKEND",     "litellm")
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "local")

# ── LiteLLM ───────────────────────────────────────────────────────────────────
LITELLM_API_KEY      = os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL     = os.getenv("LITELLM_BASE_URL", "https://litellm.example.com")
# Model identifiers are gateway-specific: a LiteLLM deployment routes names like
# "<provider>/<model>" to its own backends. Set these to whatever your gateway
# exposes; the values below are placeholders, not working defaults.
LITELLM_CHAT_MODEL   = os.getenv("LITELLM_CHAT_MODEL", "qwen3.5-35b-a3b")
LITELLM_EMBED_MODEL  = os.getenv("LITELLM_EMBED_MODEL", "qwen3-embedding-8b")
LITELLM_EMBED_DIM    = int(os.getenv("LITELLM_EMBED_DIM", "4096"))

# Optional neutral judge model for paper-metrics (avoids self-judging bias).
# Falls back to LITELLM_CHAT_MODEL when empty. Always routed via LiteLLM gateway.
JUDGE_MODEL          = os.getenv("JUDGE_MODEL", "")

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL  = "gemini-2.5-flash"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_EMBED_DIM   = 768   # gemini-embedding-001 fixed at 768

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL",  "qwen2.5:7b")
OLLAMA_EMBED_DIM   = int(os.getenv("OLLAMA_EMBED_DIM", "768"))

VLLM_CHAT_BASE_URL     = os.getenv("VLLM_CHAT_BASE_URL",     "http://localhost:8010")
VLLM_CHAT_MODEL        = os.getenv("VLLM_CHAT_MODEL",        "qwen3.6-35b-a3b")
VLLM_EMBED_BASE_URL    = os.getenv("VLLM_EMBED_BASE_URL",    "http://localhost:8011")
VLLM_EMBED_MODEL       = os.getenv("VLLM_EMBED_MODEL",       "Qwen/Qwen3-Embedding-8B")
VLLM_EMBED_DIM         = int(os.getenv("VLLM_EMBED_DIM",     "4096"))
VLLM_RERANKER_BASE_URL = os.getenv("VLLM_RERANKER_BASE_URL", "http://localhost:8012")
VLLM_RERANKER_MODEL    = os.getenv("VLLM_RERANKER_MODEL",    "BAAI/bge-reranker-v2-m3")
VLLM_API_KEY           = os.getenv("VLLM_API_KEY",           "")

_embed_dim_map = {
    "ollama":  OLLAMA_EMBED_DIM,
    "gemini":  GEMINI_EMBED_DIM,
    "litellm": LITELLM_EMBED_DIM,
    "vllm":    VLLM_EMBED_DIM,
}
EMBED_DIM = _embed_dim_map.get(EMBED_BACKEND, OLLAMA_EMBED_DIM)

WEB_SEARCH_ENABLED       = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "false"
WEB_SEARCH_PROVIDER      = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
BRAVE_SEARCH_API_KEY     = os.getenv("BRAVE_SEARCH_API_KEY", "")
TAVILY_API_KEY           = os.getenv("TAVILY_API_KEY", "")
PARALLEL_API_KEY         = os.getenv("PARALLEL_API_KEY", "")
WEB_SEARCH_MAX_RESULTS   = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_TIMEOUT_S     = int(os.getenv("WEB_SEARCH_TIMEOUT_S", "10"))
WEB_SEARCH_TRUST_PENALTY = float(os.getenv("WEB_SEARCH_TRUST_PENALTY", "0.85"))

# ── Paperless-ngx ─────────────────────────────────────────────────────────────
PAPERLESS_BASE_URL = os.getenv("PAPERLESS_BASE_URL", "http://localhost:8000/api")
PAPERLESS_TOKEN    = os.getenv("PAPERLESS_TOKEN", "")

# ── PDF Parser ────────────────────────────────────────────────────────────────
PDF_HEADING_H1_RATIO  = float(os.getenv("PDF_HEADING_H1_RATIO", "1.5"))
PDF_HEADING_H2_RATIO  = float(os.getenv("PDF_HEADING_H2_RATIO", "1.15"))
PDF_CAPTION_RATIO     = float(os.getenv("PDF_CAPTION_RATIO", "0.85"))

# ── Document Enrichment ──────────────────────────────────────────────────────
DOC_ENRICH_MAX_SECTION_INPUT = int(os.getenv("DOC_ENRICH_MAX_SECTION_INPUT", "3000"))

# ── ETL ───────────────────────────────────────────────────────────────────────
CSV_PATH         = "data/customer_support_tickets_200k.csv"
NEO4J_BATCH_SIZE = 500
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))

SAGE_ENABLED         = os.getenv("SAGE_ENABLED", "true").lower() == "true"
SAGE_TOP_K_PER_CHUNK = int(os.getenv("SAGE_TOP_K_PER_CHUNK", "5"))
SAGE_MIN_SCORE       = float(os.getenv("SAGE_MIN_SCORE", "0.88"))
SAGE_PERCENTILE      = float(os.getenv("SAGE_PERCENTILE", "95"))
SAGE_MAX_EDGES       = int(os.getenv("SAGE_MAX_EDGES", "5000"))

RETRIEVAL_TOP_K  = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANKER_TOP_K   = int(os.getenv("RERANKER_TOP_K", "10"))
RRF_K            = int(os.getenv("RRF_K", "60"))
FUSION_POOL_SIZE = int(os.getenv("FUSION_POOL_SIZE", "30"))

# ── Context-Engineering ablation switches ────────────────────────────────────
METADATA_IN_GENERATOR_ENABLED = os.getenv("METADATA_IN_GENERATOR_ENABLED", "false").lower() == "true"
METADATA_FILTER_ENABLED   = os.getenv("METADATA_FILTER_ENABLED",   "false").lower() == "true"
RERANK_BOOST_ENABLED      = os.getenv("RERANK_BOOST_ENABLED",      "false").lower() == "true"
MEMORY_ENABLED            = os.getenv("MEMORY_ENABLED",            "false").lower() == "true"
REUSE_GATE_ENABLED        = os.getenv("REUSE_GATE_ENABLED",        "false").lower() == "true"
OUTCOME_MODULATOR_ENABLED = os.getenv("OUTCOME_MODULATOR_ENABLED", "false").lower() == "true"

OUTCOME_PENALTY_PER_USER  = float(os.getenv("OUTCOME_PENALTY_PER_USER",  "0.3"))
OUTCOME_PENALTY_GLOBAL    = float(os.getenv("OUTCOME_PENALTY_GLOBAL",    "0.6"))
OUTCOME_BOOST_RESOLVED    = float(os.getenv("OUTCOME_BOOST_RESOLVED",    "1.4"))
OUTCOME_BOOST_REUSE       = float(os.getenv("OUTCOME_BOOST_REUSE",       "1.2"))
OUTCOME_SCORE_MIN         = float(os.getenv("OUTCOME_SCORE_MIN",         "0.05"))
OUTCOME_SCORE_MAX         = float(os.getenv("OUTCOME_SCORE_MAX",         "1.0"))
OUTCOME_KNOWN_ISSUE_THRESH = float(os.getenv("OUTCOME_KNOWN_ISSUE_THRESH", "0.75"))

TICKET_REUSE_THRESHOLD_HIGH = float(os.getenv("TICKET_REUSE_THRESHOLD_HIGH", "0.7"))
KI_FAST_PATH_SIM_THRESH     = float(os.getenv("KI_FAST_PATH_SIM_THRESH",     "0.85"))
TICKET_REUSE_THRESHOLD_MID  = float(os.getenv("TICKET_REUSE_THRESHOLD_MID",  "0.40"))

GRAPH_SEED_K          = int(os.getenv("GRAPH_SEED_K", "3"))
GRAPH_HOP_DEPTH       = int(os.getenv("GRAPH_HOP_DEPTH", "2"))
GRAPH_MAX_SEEDS       = int(os.getenv("GRAPH_MAX_SEEDS", "40"))
GRAPH_MIN_UNIGRAM_LEN = int(os.getenv("GRAPH_MIN_UNIGRAM_LEN", "5"))

USE_REFLACT = os.getenv("USE_REFLACT", "true").lower() == "true"
REACT_MAX_ITERATIONS = 2

# ── LLM temperatures ─────────────────────────────────────────────────────────
TEMPERATURE_STRUCTURED = float(os.getenv("TEMPERATURE_STRUCTURED", "0"))
TEMPERATURE_GENERATE   = float(os.getenv("TEMPERATURE_GENERATE",   "0.1"))
TEMPERATURE_REASON     = float(os.getenv("TEMPERATURE_REASON",     "0.3"))
TEMPERATURE_EXPAND     = float(os.getenv("TEMPERATURE_EXPAND",     "0.5"))

FASTTRACK_SIMILARITY_THRESH = float(os.getenv("FASTTRACK_SIMILARITY_THRESH", "0.82"))
FASTTRACK_HARD_SKIP_THRESH  = 0.85
FASTTRACK_SEED_THRESH       = 0.70

CRAG_SUFFICIENT_THRESH       = 0.80
CRAG_RETRY_THRESH            = 0.30
CRAG_MAX_RETRIES             = 3
CRAG_PRECISION_TOP_N         = int(os.getenv("CRAG_PRECISION_TOP_N", "5"))
CRAG_SKIP_REACT_THRESH       = 0.92
CRAG_DEFERRAL_PARTIAL_THRESH = float(os.getenv("CRAG_DEFERRAL_PARTIAL_THRESH", "0.35"))

METADATA_FILTER_KEYS = [
    "priority", "category", "product", "status", "channel", "severity",
    "doc_type", "source_system", "language",
]

# Hard keys gate retrieval (non-matching hits dropped); all others are soft boosts.
METADATA_HARD_KEYS    = set(
    k.strip() for k in os.getenv("METADATA_HARD_KEYS", "doc_type").split(",") if k.strip()
)
METADATA_BOOST_WEIGHT = float(os.getenv("METADATA_BOOST_WEIGHT", "0.15"))

# ── PIVOT ─────────────────────────────────────────────────────────────────────
PIVOT_ENABLED               = os.getenv("PIVOT_ENABLED", "true").lower() == "true"
INSPECT_REQUIRE_IMPROVEMENT = os.getenv("INSPECT_REQUIRE_IMPROVEMENT", "true").lower() == "true"
VERIFY_MIN_COVERAGE         = float(os.getenv("VERIFY_MIN_COVERAGE", "0.5"))
# When False: deferral is suppressed; pipeline produces best-effort answers instead.
ALLOW_DEFERRAL              = os.getenv("ALLOW_DEFERRAL", "true").lower() == "true"

PLAN_DEFAULT_REACT_BUDGET_FACTUAL   = int(os.getenv("PLAN_DEFAULT_REACT_BUDGET_FACTUAL",   "1"))
PLAN_DEFAULT_REACT_BUDGET_MULTI_HOP = int(os.getenv("PLAN_DEFAULT_REACT_BUDGET_MULTI_HOP", "2"))
PLAN_DEFAULT_CONFIDENCE_FACTUAL     = float(os.getenv("PLAN_DEFAULT_CONFIDENCE_FACTUAL",   "0.75"))
PLAN_DEFAULT_CONFIDENCE_MULTI_HOP   = float(os.getenv("PLAN_DEFAULT_CONFIDENCE_MULTI_HOP", "0.65"))
