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
        self.queries = query_dict
        self.textual_scores = textual_scores
        self.visual_scores = visual_scores
        self.labels = labels
        self.query_ids = query_ids

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> dict:
        query_id = self.query_ids[idx]

        return {
            "query": self.queries[query_id]["query"],
            "textual_scores": self.textual_scores[query_id],
            "visual_scores": self.visual_scores[query_id],
            "labels": self.labels[query_id],
        }
