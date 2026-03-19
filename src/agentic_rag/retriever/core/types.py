from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import torch

T = TypeVar("T", bound="RetrievalResult")


@dataclass
class RetrievalResult(ABC):
    corpus_id: int
    doc_id: str
    page_id: int
    score: float
    vector: torch.Tensor

    @classmethod
    def from_payload(cls: type[T], payload: dict, score: float, vector: torch.Tensor) -> T:
        base_kwargs = {
            "corpus_id": payload.get("corpus_id"),
            "doc_id": payload.get("doc_id"),
            "page_id": payload.get("page_id"),
            "score": score,
            "vector": vector,
        }

        extra_kwargs = cls._extra_from_payload(payload)

        return cls(**base_kwargs, **extra_kwargs)

    @classmethod
    @abstractmethod
    def _extra_from_payload(cls, payload: dict) -> dict:
        """
        Specific information from the subclasses

        Parameters
        ----------
        payload : dict
            _description_

        Returns
        -------
        dict
            _description_
        """


@dataclass
class TextualResult(RetrievalResult):
    content: str

    @classmethod
    def _extra_from_payload(cls, payload: dict) -> dict:
        return {
            "content": payload.get("content"),
        }


@dataclass
class VisualResult(RetrievalResult):
    image_path: str

    @classmethod
    def _extra_from_payload(cls, payload: dict) -> dict:
        return {
            "image_path": payload.get("image_path"),
        }


class EmbeddingType(StrEnum):
    SINGLE = "single"
    MULTI = "multi"
