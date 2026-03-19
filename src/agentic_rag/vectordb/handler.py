import logging
import uuid

import numpy as np
from qdrant_client.models import Distance, PointStruct, QueryResponse, VectorParams

from agentic_rag.vectordb.connection import QdrantSingleton, QdrantSingletonFactory
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class QdrantHandler:
    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        singleton: QdrantSingleton = QdrantSingletonFactory.get_instance(host, port)
        self.client = singleton.client

    def create_collection(self, name: str, dim: int) -> None:
        try:
            existing = [c.name for c in self.client.get_collections().collections]

            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{name}' created")
        except Exception as e:
            logger.error(f"Error while creating collection '{name}': {e}")

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)
        logger.info(f"Collection '{name}' deleted")

    def add(
        self,
        collection_name: str,
        vectors: list[np.ndarray],
        payloads: list[dict[str,]],
    ) -> None:
        points = []

        for vec, payload in zip(vectors, payloads):
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()

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

        logger.info(f"Inserted {len(points)} points into '{collection_name}'")

    def search(self, collection_name: str, query_vector: np.ndarray, k: int = 5) -> QueryResponse:
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
        )
