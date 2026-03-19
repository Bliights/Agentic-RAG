from abc import ABC
from dataclasses import dataclass


@dataclass
class RetrievalResult(ABC):
    doc_id: int
    page_id: int
    score: float


@dataclass
class TextualResult(RetrievalResult):
    text: str


@dataclass
class VisualResult(RetrievalResult):
    image_path: str
