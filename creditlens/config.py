"""Central configuration.

All runtime knobs live here so that behaviour is reproducible and auditable:
an analysis run records the settings fingerprint it was produced under.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CREDITLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- environment -------------------------------------------------------
    env: Literal["dev", "test", "prod"] = "dev"
    debug: bool = False

    # --- storage -----------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    database_url: str = ""  # derived below if empty

    # --- LLM ---------------------------------------------------------------
    # `auto` picks the first provider whose credentials actually resolve, in
    # the order declared by creditlens.agent.providers.AUTO_ORDER, and falls
    # back to the deterministic engine when none do. Name a provider to pin it.
    #
    # Credentials are never read directly: an unset ANTHROPIC_API_KEY does not
    # mean "no credentials" (the SDK also resolves ANTHROPIC_AUTH_TOKEN and
    # `ant auth login` profiles), so construction is attempted and failure is
    # what triggers the fallback.
    llm_provider: str = "auto"
    #: empty means "use the selected provider's default model"
    model: str = ""
    fast_model: str = ""
    max_tokens: int = 8000
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    thinking: bool = True
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 2

    # --- provider-specific ---------------------------------------------------
    aws_region: str = "us-east-1"
    gcp_project: str = ""
    gcp_region: str = "global"

    # --- agent budgets -----------------------------------------------------
    max_tool_iterations: int = 8
    max_tool_calls: int = 24
    agent_token_budget: int = 250_000

    # --- retrieval ---------------------------------------------------------
    # `hashed` needs nothing and is reproducible in CI; `ollama` runs a real
    # model locally with no key and no data egress. Changing either the provider
    # or the model changes the vector space, so stored vectors must be rebuilt
    # with `creditlens reembed`.
    embedding_provider: str = "hashed"
    #: empty means "the provider's default model"
    embedding_model: str = ""
    embedding_dim: int = 512
    embedding_batch_size: int = 64
    chunk_target_tokens: int = 380
    chunk_overlap_tokens: int = 60
    retrieval_top_k: int = 8
    retrieval_candidate_k: int = 40
    rrf_k: int = 60
    # Defaults below are set from scripts/sweep_retrieval.py, not by feel.
    # MMR cost 0.24 nDCG@8 on the labelled set (0.68 -> 0.44 at lambda 0.7):
    # relevant passages in filings genuinely cluster, so diversity trades away
    # hits. Off by default, still available per request.
    retrieval_use_mmr: bool = False
    mmr_lambda: float = 0.95
    # 0.2 is the best paraphrase-query setting (nDCG 0.115 vs 0.092 lexical-only)
    # at a cost of 0.013 nDCG on keyword queries.
    dense_weight: float = 0.2
    #: minimum passages guaranteed per issuer when a query scopes several
    per_issuer_floor: int = 3

    # --- ingestion ---------------------------------------------------------
    edgar_user_agent: str = "CreditLens/0.1 (contact@example.com)"
    edgar_base_url: str = "https://data.sec.gov"
    edgar_www_url: str = "https://www.sec.gov"
    edgar_rate_limit_per_s: float = 8.0
    http_timeout_s: float = 30.0

    # --- verification ------------------------------------------------------
    numeric_tolerance_pct: float = 0.5  # |claim - computed| / |computed| * 100
    min_citations_for_high_confidence: int = 3

    # --- api ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "creditlens.db"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.sqlite_path}"

    def fingerprint(self) -> str:
        """Stable hash of the knobs that can change an answer.

        Recorded on every analysis so results can be reproduced or invalidated.
        """
        material = {
            "provider": self.llm_provider,
            "model": self.model,
            "effort": self.effort,
            "thinking": self.thinking,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "chunk_target_tokens": self.chunk_target_tokens,
            "retrieval_top_k": self.retrieval_top_k,
            "dense_weight": self.dense_weight,
            "rrf_k": self.rrf_k,
            "retrieval_use_mmr": self.retrieval_use_mmr,
            "numeric_tolerance_pct": self.numeric_tolerance_pct,
        }
        blob = json.dumps(material, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
