from pathlib import Path

DATA_FOLDER = Path(__file__).resolve().parents[4] / "data"
DB_HOST = "localhost"
DB_PORT = 6333
TOKENISER_MODEL_NAME = "cl100k_base"
EMBEDDER_MODEL_NAME = "intfloat/multilingual-e5-large"
DATABASE_NAME = "textual"
EMBEDDING_DIM = 1024
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
