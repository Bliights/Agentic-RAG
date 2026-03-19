from abc import ABC, abstractmethod

from src.retriever.core.types import BaseScore


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[BaseScore]:
        pass
