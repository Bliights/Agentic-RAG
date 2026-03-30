from dataclasses import dataclass
from enum import StrEnum


@dataclass
class HybridResult:
    corpus_id: int
    doc_id: str
    page_id: int
    score: float

    chunk_id: int
    content: str
    image_path: str

    textual_rank: int | None = None
    visual_rank: int | None = None


class RetrieverMode(StrEnum):
    RRF = "RRF"
    ALPHA = "alpha"


@dataclass
class PipelineAnswer:
    answer: str
    docs: list[HybridResult]
