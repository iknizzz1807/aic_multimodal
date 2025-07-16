from typing import List, Optional
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    text: str = Field(
        ...,
        description="Natural language description",
        example="a person walking on the beach",
    )
    top_k: int = Field(
        default=12, ge=1, le=100, description="Number of results (1-100)"
    )


class VisualResult(BaseModel):
    id: str
    score: float
    timestamp: Optional[float] = None
    image: Optional[str] = None


class AudioResult(BaseModel):
    id: str
    score: float
    match: str
    start: float
    end: float


class MatchDetail(BaseModel):
    type: str
    timestamp: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text_match: Optional[str] = None
    score: float


class UnifiedResult(BaseModel):
    id: str
    score: float
    reason: List[str]
    image: Optional[str] = None
    details: List[MatchDetail]


class VisualSearchResponse(BaseModel):
    results: List[VisualResult]


class AudioSearchResponse(BaseModel):
    results: List[AudioResult]


class UnifiedSearchResponse(BaseModel):
    results: List[UnifiedResult]
