import numpy as np
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image

from agentic_rag.retriever.core.base import BaseEmbedder


class VisualEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "vidore/colqwen2-v0.1") -> None:
        """
        Initialize a visual embedder

        Parameters
        ----------
        model_name : str, optional
            Name or path of the pretrained ColQwen2 model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.model.eval()

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a textual query into an embedding

        Parameters
        ----------
        query : str
            Input query

        Returns
        -------
        torch.Tensor
            Embedding tensor representing the query
        """
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            return self.model(**batch)

    def encode_image(self, image: Image) -> torch.Tensor:
        """
        Encode an image into an embedding

        Parameters
        ----------
        image : Image
            Input image

        Returns
        -------
        torch.Tensor
            Embedding tensor representing the image
        """
        batch = self.processor.process_images([image]).to(self.device)
        with torch.no_grad():
            return self.model(**batch)

    def to_numpy(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Convert a torch embedding tensor to a NumPy array

        Parameters
        ----------
        embeddings : torch.Tensor
            Embedding tensor

        Returns
        -------
        np.ndarray
            NumPy representation
        """
        embeddings = embeddings.detach().cpu()
        return embeddings[0].numpy()
