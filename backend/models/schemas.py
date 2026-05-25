from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5


class QueryIntent(BaseModel):
    genes: Optional[list[str]] = None
    genotypes: Optional[dict[str, str]] = None
    drug: Optional[str] = None
    drug_class: Optional[str] = None
    disease: Optional[str] = None
    symptom: Optional[str] = None
    condition: Optional[str] = None
    intent: Optional[str] = None


class SourceRef(BaseModel):
    id: int
    content: str
    source: str
    year: Optional[str] = None
    drug: Optional[str] = None
    gene: Optional[str] = None
    relevance: Optional[float] = None


class ReportResponse(BaseModel):
    query: str
    parsed_intent: Optional[QueryIntent] = None
    rule_engine_result: Optional[str] = None
    report_text: str
    sources: list[SourceRef]
