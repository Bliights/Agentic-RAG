from abc import ABC, abstractmethod

from agentic_rag.retriever.core.types import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        pass
