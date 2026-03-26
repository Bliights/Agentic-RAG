from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T", bound="RetrievalResult")


@dataclass
class RetrievalResult(ABC):
    corpus_id: int
    doc_id: str
    page_id: int
    score: float
    vector: torch.Tensor | np.ndarray

    @classmethod
    def from_payload(
        cls: type[T],
        payload: dict,
        score: float,
        vector: torch.Tensor | np.ndarray,
    ) -> T:
        """
        Build a retrieval result instance from a payload dictionary

        Parameters
        ----------
        cls : type[T]
            Concrete subclass of RetrievalResult to instantiate
        payload : dict
            Payload containing metadata
        score : float
            Similarity or relevance score
        vector : torch.Tensor | np.ndarray
            Embedding vector

        Returns
        -------
        T
            Instantiated retrieval result object
        """
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
        Extract subclass-specific fields from a payload dictionary

        Parameters
        ----------
        payload : dict
            Payload containing metadata

        Returns
        -------
        dict
            Dictionary of keyword arguments required to instantiate the
            subclass-specific fields
        """


@dataclass
class TextualResult(RetrievalResult):
    chunk_id: int
    content: str

    @classmethod
    def _extra_from_payload(cls, payload: dict) -> dict:
        """
        Extract subclass-specific fields from a payload dictionary

        Parameters
        ----------
        payload : dict
            Payload containing metadata

        Returns
        -------
        dict
            Dictionary containing the textual chunk identifier and content
        """
        return {
            "chunk_id": payload.get("chunk_id"),
            "content": payload.get("content"),
        }


@dataclass
class VisualResult(RetrievalResult):
    image_path: str

    @classmethod
    def _extra_from_payload(cls, payload: dict) -> dict:
        """
        Extract subclass-specific fields from a payload dictionary

        Parameters
        ----------
        payload : dict
            Payload containing metadata

        Returns
        -------
        dict
            Dictionary containing the image path
        """
        return {
            "image_path": payload.get("image_path"),
        }


class EmbeddingType(StrEnum):
    SINGLE = "single"
    MULTI = "multi"
