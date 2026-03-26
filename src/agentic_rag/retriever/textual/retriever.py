import numpy as np

from agentic_rag.retriever.core.base import BaseRetriever
from agentic_rag.retriever.core.types import TextualResult
from agentic_rag.retriever.textual.embedder import TextualEmbedder
from agentic_rag.vectordb.handler import QdrantHandler


class TextualRetriever(BaseRetriever):
    def __init__(self, collection_name: str, db_handler: QdrantHandler, embedder_name: str) -> None:
        """
        Initialize a textual retriever backed by a Qdrant vector database

        Parameters
        ----------
        collection_name : str
            Name of the Qdrant collection storing textual embeddings
        db_handler : QdrantHandler
            Handler responsible for communicating with the Qdrant database
        embedder_name : str
            Name or identifier of the embedding model
        """
        self.collection_name = collection_name
        self.db = db_handler
        self.embedder = TextualEmbedder(embedder_name)

    def search(self, query: str, k: int = 5) -> list[TextualResult]:
        """
        Retrieve the top-k most relevant textual results for a given query

        Parameters
        ----------
        query : str
            Input query
        k : int, optional
            Number of results to retrieve

        Returns
        -------
        list[TextualResult]
            Ranked list of textual retrieval results
        """
        query_vector = self.embedder.encode_query(query)

        matching = self.db.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            k=k,
        )

        return [
            TextualResult.from_payload(
                r.payload,
                score=r.score,
                vector=np.array(r.vector, dtype=np.float32),
            )
            for r in matching.points
        ]
