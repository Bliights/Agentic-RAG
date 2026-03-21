from abc import ABC, abstractmethod

import numpy as np
import torch

from agentic_rag.retriever.core.types import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        pass


class BaseEmbedder(ABC):
    @abstractmethod
    def encode_query(self, query: str) -> torch.Tensor | np.ndarray:
        pass
