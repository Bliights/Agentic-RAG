import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from agentic_rag.utils.metrics import ndcg_at_k, recall_at_k


class ScorerModel(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 256,
        metrics_nb: int = 5,
    ) -> None:
        """
        Initialize the scorer model used to predict the weight alpha from a query

        Parameters
        ----------
        model_name : str, optional
            Name of the pretrained transformer model
        hidden_dim : int, optional
            Hidden dimension of the scoring MLP
        metrics_nb : int, optional
            Cutoff value used for ranking metrics
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.metrics_nb = metrics_nb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        h = self.encoder.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(h, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def tokenize(self, queries: list[str]) -> BatchEncoding:
        """
        Tokenize a batch of query strings

        Parameters
        ----------
        queries : list[str]
            List of query strings to tokenize

        Returns
        -------
        BatchEncoding
            Tokenized queries
        """
        return self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha coefficients for a batch of tokenized queries

        Parameters
        ----------
        **kwargs : torch.Tensor
            Tokenized query inputs

        Returns
        -------
        torch.Tensor
            Predicted alpha coefficients
        """
        outputs = self.encoder(**kwargs)
        cls = outputs.last_hidden_state[:, 0]
        return self.mlp(cls).squeeze(-1)

    def predict(self, queries: list[str]) -> torch.Tensor:
        """
        Predict alpha coefficients for a list of raw query

        Parameters
        ----------
        queries : list[str]
            List of input query

        Returns
        -------
        torch.Tensor
            Predicted alpha coefficients
        """
        self.eval()
        tokens = self.tokenize(queries)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            return self(**tokens)

    def _pairwise_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """
        Compute a pairwise ranking loss over a batch of predicted scores

        Parameters
        ----------
        scores : torch.Tensor
            Predicted fused scores for retrieved items
        labels : torch.Tensor
            Ground-truth labels for retrieved items
        margin : float
            Margin applied in the pairwise ranking objective

        Returns
        -------
        torch.Tensor
            Loss value averaged over all valid item pairs in the batch
        """
        losses = []

        for b in range(labels.shape[0]):
            y = labels[b]
            s = scores[b]

            sorted_idx = torch.argsort(y, descending=True)

            for i in range(len(sorted_idx)):
                for j in range(i + 1, len(sorted_idx)):
                    yi = y[sorted_idx[i]]
                    yj = y[sorted_idx[j]]

                    if yi > yj:
                        si = s[sorted_idx[i]]
                        sj = s[sorted_idx[j]]

                        weight = 2**yi - 2**yj

                        loss = weight * nn.functional.softplus(-(si - sj - margin))
                        losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return torch.stack(losses).mean()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float = 2e-5,
        margin: float = 0.1,
    ) -> None:
        """
        Train the scorer model and keep the best checkpoint based on validation NDCG

        Parameters
        ----------
        train_loader : DataLoader
            Training dataset
        val_loader : DataLoader
            Validation dataset
        epochs : int
            Number of training epochs
        lr : float, optional
            Learning rate
        margin : float, optional
            Margin used in the pairwise ranking loss
        """
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        best_ndcg = -float("inf")
        best_weights = copy.deepcopy(self.state_dict())

        with tqdm(
            range(epochs),
            desc="Training Epochs",
            colour="green",
        ) as epoch_bar:
            for _ in epoch_bar:
                train_loss = self._train_one_epoch(train_loader, optimizer, margin)
                val_loss, val_metrics = self._evaluate(val_loader, margin, verbose=False)

                epoch_bar.set_postfix_str(
                    f"train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"ndcg@{self.metrics_nb}={val_metrics[f'ndcg@{self.metrics_nb}']:.3f}  "
                    f"recall@{self.metrics_nb}={val_metrics[f'recall@{self.metrics_nb}']:.3f}",
                )

                if best_ndcg < val_metrics[f"ndcg@{self.metrics_nb}"]:
                    best_ndcg = val_metrics[f"ndcg@{self.metrics_nb}"]
                    best_weights = copy.deepcopy(self.state_dict())

        self.load_state_dict(best_weights)

    def _train_one_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        margin: float,
    ) -> float:
        """
        Train the model for one epoch

        Parameters
        ----------
        dataloader : DataLoader
            Dataset provided
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters
        margin : float
            Margin used in the pairwise ranking loss

        Returns
        -------
        float
            Average training loss over the epoch
        """
        self.train()
        total_loss = 0.0

        with tqdm(
            dataloader,
            desc="Training Batches",
            bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
            colour="yellow",
            leave=False,
        ) as batch_bar:
            for batch in batch_bar:
                query = batch["query"]
                textual_scores = batch["textual_scores"].to(self.device)
                visual_scores = batch["visual_scores"].to(self.device)
                labels = batch["labels"].to(self.device)

                tokens = self.tokenize(query)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                optimizer.zero_grad()
                alpha = self(**tokens)
                scores = (
                    alpha.unsqueeze(1) * textual_scores + (1 - alpha).unsqueeze(1) * visual_scores
                )

                loss = self._pairwise_loss(scores, labels, margin)

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

                batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(dataloader)

    def _evaluate(
        self,
        dataloader: DataLoader,
        margin: float,
        verbose: bool = False,
    ) -> float:
        """
        Evaluate the model

        Parameters
        ----------
        dataloader : DataLoader
            Dataset used for evaluation
        margin : float
            Margin used in the pairwise ranking loss
        verbose : bool, optional
            Whether to display a progress bar during evaluation

        Returns
        -------
        float
            the average loss and a dictionary of ranking metrics computed over the dataset
        """
        self.eval()
        total_loss = 0.0
        ndcg_scores = []
        recall_scores = []

        iterator = (
            tqdm(
                dataloader,
                desc="Evaluating",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
                colour="magenta",
            )
            if verbose
            else dataloader
        )
        with torch.no_grad():
            for batch in iterator:
                query = batch["query"]
                textual_scores = batch["textual_scores"].to(self.device)
                visual_scores = batch["visual_scores"].to(self.device)
                labels = batch["labels"].to(self.device)

                tokens = self.tokenize(query)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                alpha = self(**tokens)
                scores = (
                    alpha.unsqueeze(1) * textual_scores + (1 - alpha).unsqueeze(1) * visual_scores
                )

                loss = self._pairwise_loss(scores, labels, margin)
                total_loss += float(loss.item())

                for i in range(scores.shape[0]):
                    pred = scores[i].cpu().numpy()
                    y = labels[i].cpu().numpy()

                    ndcg_scores.append(ndcg_at_k(pred, y, self.metrics_nb))
                    recall_scores.append(recall_at_k(pred, y, self.metrics_nb))

                if verbose:
                    iterator.set_postfix_str(f"loss={loss.item():.4f}")

        metrics = {
            f"ndcg@{self.metrics_nb}": float(np.mean(ndcg_scores)),
            f"recall@{self.metrics_nb}": float(np.mean(recall_scores)),
        }

        return total_loss / len(dataloader), metrics

    def save(self, path: str) -> None:
        """
        Save the model configuration and weights to disk

        Parameters
        ----------
        path : str
            Destination file path

        Raises
        ------
        ValueError
            If the parent directory of path does not exist
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            raise ValueError(f"Directory '{directory}' does not exist.")
        state = {
            "config": {
                "model_name": self.model_name,
                "hidden_dim": self.hidden_dim,
            },
            "model_state_dict": self.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device | None = None,
    ) -> "ScorerModel":
        """
        Load a scorer model from disk

        Parameters
        ----------
        path : str
            Path to the saved model
        device : torch.device | None, optional
            Device on which to load the model

        Returns
        -------
        ScorerModel
            Loaded scorer model in evaluation mode

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        RuntimeError
            If the file cannot be loaded correctly
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file '{path}' does not exist.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            checkpoint = torch.load(path, map_location=device)

            config = checkpoint["config"]

            model = cls(
                model_name=config["model_name"],
                hidden_dim=config["hidden_dim"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            return model

        except RuntimeError as e:
            raise RuntimeError(f"Failed to load model state : {e}")
