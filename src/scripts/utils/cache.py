import json
from pathlib import Path
from typing import TypeVar

import pandas as pd
import torch

T = TypeVar("T")


class CacheManager:
    """
    Utility class to save and load objects
    """

    @classmethod
    def save(cls, obj: T, path: str) -> None:
        """
        Save an object to disk

        Parameters
        ----------
        obj : T
            Object to save
        path : str
            Destination file path

        Raises
        ------
        TypeError
            If the object type does not match the required format
        ValueError
            If the file extension is not supported
        """
        path: Path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix

        if ext == ".txt":
            if not isinstance(obj, str):
                raise TypeError("Only strings can be saved as .txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(obj)
        elif ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
        elif ext == ".pt":
            torch.save(obj, path)
        elif ext == ".csv":
            if not isinstance(obj, pd.DataFrame):
                raise TypeError("Only pandas DataFrame can be saved as .csv")
            obj.to_csv(path, sep="|", index=False)
        else:
            raise ValueError(f"Unsupported cache format: {ext}")

    @classmethod
    def load(cls, path: str) -> T:
        """
        Load an object from disk

        Parameters
        ----------
        path : str
            Path to the cached object

        Returns
        -------
        T
            Loaded object

        Raises
        ------
        FileNotFoundError
            If the specified cache file does not exist
        ValueError
            If the file extension is not supported
        """
        path: Path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Cache file '{path}' not found")

        ext = path.suffix

        if ext == ".txt":
            with open(path, encoding="utf-8") as f:
                return f.read().replace("\n", " ")
        elif ext == ".json":
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        elif ext == ".pt":
            return torch.load(path, weights_only=False)
        elif ext == ".csv":
            return pd.read_csv(path, sep="|")
        else:
            raise ValueError(f"Unsupported cache format: {ext}")
