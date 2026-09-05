"""Request/response models.

Pydantic models exist at the boundary only. Internal code passes dataclasses;
converting once at the edge keeps validation errors close to the caller and
keeps the analysis core independent of the web framework.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000,
                          description="The credit question to answer.")
    tickers: list[str] | None = Field(
        default=None, max_length=6,
        description="Optional scope. Inferred from the question when omitted.",
    )
    include_trace: bool = Field(
        default=False, description="Include the full span trace in the response."
    )

    @field_validator("tickers")
    @classmethod
    def _upper(cls, value: list[str] | None) -> list[str] | None:
        return [t.strip().upper() for t in value if t.strip()] if value else None


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    tickers: list[str] | None = None
    form_types: list[str] | None = None
    items: list[str] | None = None
    fiscal_years: list[int] | None = None
    top_k: int = Field(default=8, ge=1, le=25)
    include_text: bool = False


class IngestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=8)
    max_filings: int = Field(default=6, ge=1, le=20)
    forms: list[Literal["10-K", "10-Q"]] = Field(default_factory=lambda: ["10-K", "10-Q"])
    min_fiscal_year: int | None = Field(default=None, ge=1990, le=2100)


class CompareRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=6)
    period: str | None = None
    ratios: list[str] | None = None


class EvalRequest(BaseModel):
    suite: str = "golden"
    limit: int | None = Field(default=None, ge=1, le=200)
    engine: Literal["auto", "offline"] = "auto"


class HealthResponse(BaseModel):
    status: str
    version: str
    database: bool
    corpus: dict[str, Any]
    llm: dict[str, Any]
    settings_fingerprint: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
