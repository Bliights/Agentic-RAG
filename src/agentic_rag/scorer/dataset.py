from torch.utils.data import Dataset


class ScorerDataset(Dataset):
    def __init__(
        self,
        query_dict: dict,
        textual_scores: dict,
        visual_scores: dict,
        labels: dict,
        query_ids: list[int],
    ) -> None:
        """
        Initialize a dataset for training or evaluating the scorer model

        Parameters
        ----------
        query_dict : dict
            Mapping from query IDs to query metadata
        textual_scores : dict
            Mapping from query IDs to textual retrieval scores
        visual_scores : dict
            Mapping from query IDs to visual retrieval scores
        labels : dict
            Mapping from query IDs to labels
        query_ids : list[int]
            Ordered list of query IDs to include in the dataset
        """
        self.queries = query_dict
        self.textual_scores = textual_scores
        self.visual_scores = visual_scores
        self.labels = labels
        self.query_ids = query_ids

    def __len__(self) -> int:
        """
        Return the number of queries in the dataset

        Returns
        -------
        int
            Number of dataset items
        """
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a dataset item by index

        Parameters
        ----------
        idx : int
            Index of the item to retrieve

        Returns
        -------
        dict
            Dictionary containing the query text, textual scores, visual
            scores, and relevance labels for the selected query
        """
        query_id = self.query_ids[idx]

        return {
            "query": self.queries[query_id]["query"],
            "textual_scores": self.textual_scores[query_id],
            "visual_scores": self.visual_scores[query_id],
            "labels": self.labels[query_id],
        }
