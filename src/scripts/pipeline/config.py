from pathlib import Path

from agentic_rag.pipeline.types import RetrieverMode

DATA_FOLDER = Path(__file__).resolve().parents[3] / "data"
DB_HOST = "localhost"
DB_PORT = 6333
METRICS_NB_DOC = 5
NB_RETRIEVE = METRICS_NB_DOC * 3
MODE = RetrieverMode.ALPHA
SCORER_PATH = Path(__file__).resolve().parents[3] / "models" / "scorer_model.pth"
LLM_MODEL_NAME = "ministral-3:3b"  # "qwen3.5:0.8b"
