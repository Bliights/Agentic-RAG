import torch

from agentic_rag.scorer.model import ScorerModel


class Scorer:
    def __init__(
        self,
        model_path: str,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize a scorer used to compute the alpha for the combine scores

        Parameters
        ----------
        model_path : str
            Path to the saved scorer
        device : torch.device | None, optional
            Device on which to load the model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScorerModel.load(model_path, device)

    def compute_alpha(self, query: str) -> float:
        """
        Compute the alpha coefficient for a given query

        Parameters
        ----------
        query : str
            Input query

        Returns
        -------
        float
            Predicted alpha coefficient
        """
        return self.model.predict([query]).item()

    def fuse_scores(
        self,
        alpha: float,
        textual_score: float,
        visual_score: float,
    ) -> float:
        """
        Combine textual and visual scores

        Parameters
        ----------
        alpha : float
            Alpha weight used in the scoring
        textual_score : float
            Score obtained from the textual retriever
        visual_score : float
            Score obtained from the visual retriever

        Returns
        -------
        float
            Fused score computed
        """
        return alpha * textual_score + (1 - alpha) * visual_score
