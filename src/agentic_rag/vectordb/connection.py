from qdrant_client import QdrantClient


class QdrantSingleton:
    _instance = None
    _client: QdrantClient | None = None

    def __new__(cls, host: str = "localhost", port: int = 6333) -> "QdrantSingleton":
        """
        Create or return a singleton instance of the Qdrant client wrapper

        Parameters
        ----------
        host : str, optional
            Hostname of the Qdrant server
        port : int, optional
            Port of the Qdrant server

        Returns
        -------
        QdrantSingleton
            Singleton instance managing the Qdrant client
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(host, port)
        return cls._instance

    def initialize(self, host: str, port: int) -> None:
        """
        Initialize the Qdrant client if not already created

        Parameters
        ----------
        host : str, optional
            Hostname of the Qdrant server
        port : int, optional
            Port of the Qdrant server
        """
        if not self._client:
            self._client = QdrantClient(host=host, port=port)

    @property
    def client(self) -> QdrantClient:
        """
        Access the underlying Qdrant client instance

        Returns
        -------
        QdrantClient
            Initialized Qdrant client
        """
        return self._client


class QdrantSingletonFactory:
    _instances = {}

    @classmethod
    def get_instance(cls, host: str = "localhost", port: int = 6333) -> QdrantSingleton:
        """
        Retrieve or create a QdrantSingleton instance for a given host and port

        Parameters
        ----------
        host : str, optional
            Hostname of the Qdrant server
        port : int, optional
            Port of the Qdrant server

        Returns
        -------
        QdrantSingleton
            Singleton instance corresponding to the specified connection
        """
        key = f"{host}:{port}"

        if key not in cls._instances:
            cls._instances[key] = QdrantSingleton(host, port)

        return cls._instances[key]
