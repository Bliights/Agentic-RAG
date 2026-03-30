from pathlib import Path

DATA_FOLDER = Path(__file__).resolve().parents[3] / "data"
BATCH_SIZE = 64
DB_HOST = "localhost"
DB_PORT = 6333
TEXTUAL_DATABASE_NAME = "textual"
VISUAL_DATABASE_NAME = "visual"
TEST_RATIO = 0.2
VAL_RATIO = 0.2
HIDDEN_DIM = 256
TRANSFORMER_NAME = "distilbert-base-uncased"
EPOCHS = 20
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "scorer_model.pth"

METRICS_NB_DOC = 5
NB_DOCS_RETRIEVAL = 20
TOTAL_MAX_DOC_RETRIEVAL = 40
FEATURE_CACHE_PATH = (
    Path(__file__).resolve().parents[3]
    / ".cache"
    / f"features_k{NB_DOCS_RETRIEVAL}_maxdoc{TOTAL_MAX_DOC_RETRIEVAL}.pt"
)
