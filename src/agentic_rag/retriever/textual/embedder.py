import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from agentic_rag.retriever.core.base import BaseEmbedder


class TextualEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
    ) -> None:
        """
        Initialize a textual embedder using a SentenceTransformer model

        Parameters
        ----------
        model_name : str, optional
            Name of the SentenceTransformer model to load
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query string into a normalized embedding vector

        Parameters
        ----------
        query : str
            Input query

        Returns
        -------
        np.ndarray
            Normalized embedding vector representing the query
        """
        query = f"query: {query}"
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0]

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        """
        Encode a list of documents into normalized embedding vectors

        Parameters
        ----------
        texts : list[str]
            List of input texts to encode

        Returns
        -------
        list[np.ndarray]
            List of normalized embedding vectors
        """
        texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return list(embeddings)
