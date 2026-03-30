from pathlib import Path

DATA_FOLDER = Path(__file__).resolve().parents[3] / "data"
DB_HOST = "localhost"
DB_PORT = 6333
SCORER_PATH = Path(__file__).resolve().parents[3] / "models" / "scorer_model.pth"
LLM_MODEL_NAME = "ministral-3:3b"  # "qwen3.5:0.8b"
