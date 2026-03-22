import torch

from agentic_rag.retriever.core.types import TextualResult, VisualResult
from agentic_rag.scorer.model import ScorerModel


class Scorer:
    def __init__(
        self,
        model_path: str,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScorerModel.load(model_path, device)

    def get_score(
        self,
        query: str,
        textual_result: TextualResult,
        visual_result: VisualResult,
    ) -> float:
        alpha = self.model.predict([query]).item()
        return alpha * textual_result.score + (1 - alpha) * visual_result.score
