from qdrant_client import QdrantClient


class QdrantSingleton:
    _instance = None
    _client: QdrantClient | None = None

    def __new__(cls, host: str = "localhost", port: int = 6333) -> "QdrantSingleton":
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(host, port)
        return cls._instance

    def initialize(self, host: str, port: int) -> None:
        if not self._client:
            self._client = QdrantClient(host=host, port=port)

    @property
    def client(self) -> QdrantClient:
        return self._client


class QdrantSingletonFactory:
    _instances = {}

    @classmethod
    def get_instance(cls, host: str = "localhost", port: int = 6333) -> QdrantSingleton:
        key = f"{host}:{port}"

        if key not in cls._instances:
            cls._instances[key] = QdrantSingleton(host, port)

        return cls._instances[key]
