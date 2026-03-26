import logging
import uuid

import numpy as np
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    QueryResponse,
    VectorParams,
)

from agentic_rag.retriever.core.types import EmbeddingType
from agentic_rag.vectordb.connection import QdrantSingleton, QdrantSingletonFactory

logger = logging.getLogger(__name__)


class QdrantHandler:
    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        """
        Initialize a handler for interacting with a Qdrant vector database

        Parameters
        ----------
        host : str, optional
            Hostname of the Qdrant server
        port : int, optional
            Port of the Qdrant server
        """
        singleton: QdrantSingleton = QdrantSingletonFactory.get_instance(host, port)
        self.client = singleton.client

    def create_collection(self, name: str, embedding_type: EmbeddingType, dim: int) -> None:
        """
        Create a Qdrant collection with the specified embedding configuration

        Parameters
        ----------
        name : str
            Name of the collection to create
        embedding_type : EmbeddingType
            Type of embeddings used
        dim : int
            Dimensionality of the embedding vectors

        Raises
        ------
        ValueError
            If an unsupported embedding type is provided
        """
        try:
            existing = [c.name for c in self.client.get_collections().collections]

            if name in existing:
                logger.info(f"Collection '{name}' already exists")
                return

            if embedding_type == EmbeddingType.SINGLE:
                vectors_config = VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                )
            elif embedding_type == EmbeddingType.MULTI:
                vectors_config = VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=HnswConfigDiff(m=0),
                )
            else:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")

            self.client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
            )
            logger.info(f"Collection '{name}' created")
        except Exception as e:
            logger.error(f"Error while creating collection '{name}': {e}")

    def delete_collection(self, name: str, verbose: bool = True) -> None:
        """
        Delete a collection from the Qdrant database

        Parameters
        ----------
        name : str
            Name of the collection to delete
        verbose : bool, optional
            Whether to log the deletion operation
        """
        self.client.delete_collection(name)
        if verbose:
            logger.info(f"Collection '{name}' deleted")

    def add(
        self,
        collection_name: str,
        vectors: list[np.ndarray],
        payloads: list[dict[str,]],
        verbose: bool = True,
    ) -> None:
        """
        Insert vectors and their associated payloads into a collection

        Parameters
        ----------
        collection_name : str
            Name of the target collection
        vectors : list[np.ndarray]
            List of vectors or multi-vectors to insert
        payloads : list[dict[str,]]
            List of payload dictionaries associated with each vector
        verbose : bool, optional
            Whether to log the insertion operation
        """
        points = []
        for vec, payload in zip(vectors, payloads):
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            if isinstance(vec, list) and isinstance(vec[0], np.ndarray):
                vec = [v.tolist() for v in vec]
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                ),
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )
        if verbose:
            logger.info(f"Inserted {len(points)} points into '{collection_name}'")

    def search(self, collection_name: str, query_vector: np.ndarray, k: int = 5) -> QueryResponse:
        """
        Perform a similarity search in a Qdrant collection

        Parameters
        ----------
        collection_name : str
            Name of the collection to query
        query_vector : np.ndarray
            Query embedding vector or multi-vector
        k : int, optional
            Number of results to retrieve

        Returns
        -------
        QueryResponse
            Response containing the top-k nearest neighbors and associated metadata
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        if isinstance(query_vector, list) and isinstance(query_vector[0], np.ndarray):
            query_vector = [v.tolist() for v in query_vector]

        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            with_vectors=True,
        )
