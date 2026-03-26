from abc import ABC, abstractmethod

import numpy as np
import torch

from agentic_rag.retriever.core.types import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """
        Search the retrieval index for the most relevant results matching a query

        Parameters
        ----------
        query : str
            Input query used to retrieve relevant items
        k : int, optional
            Maximum number of results to return

        Returns
        -------
        list[RetrievalResult]
            Ranked list of retrieval results matching the query
        """


class BaseEmbedder(ABC):
    @abstractmethod
    def encode_query(self, query: str) -> torch.Tensor | np.ndarray:
        """
        Encode a query string into a vector representation

        Parameters
        ----------
        query : str
            Input query to encode

        Returns
        -------
        torch.Tensor | np.ndarray
            Vector representation of the query
        """
