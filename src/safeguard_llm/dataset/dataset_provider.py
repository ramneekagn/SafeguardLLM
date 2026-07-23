from typing import Tuple, Any, List
from datasets import load_dataset, concatenate_datasets

class DatasetProvider():
    """ A provider class for loading, shuffling, and sampling datasets from HuggingFace.

    Attributes:
        dataset_name: The id to the dataset e.g. walledai/XSTest
        dataset_source: Source plattform to search the dataset from. Supports "hf" - huggingface

    """
    def __init__(self, dataset_name: str, source: str = "hf") -> None:
        """Initializes the DatasetProvider class.

        Args:
            dataset_name: The id to the dataset to load
            source: Source plattform to search the dataset. Supports "hf" - Huggingface. Defaults to "hf" - Huggingface
        """
        self.dataset_name = dataset_name
        self.dataset_source = source

    def get_dataset(self, split: str, seed: int, subset: str, size: int) -> Tuple[List[str], List[int]]:
        """ Loads and samples a subset from a specific data split.

        Args:
            split: Dataset split to load on such as 'train', 'eval'
            seed: Random seed for reproduceable shuffling for the dataset
            subset: Specific subset of the dataset to select from
            size: Maximum number of rows to use from the dataset.

        Returns:
            Tuple[List[str], List[int]]: A standard HUggingface dataset object.

        Raises:
            ValueError: If 'dataset_source' is not supported.

        """
        if self.dataset_source == "hf":
            from datasets import load_dataset
            ds = load_dataset(self.dataset_name,split)            
            ds = ds.shuffle(seed)
            ds = ds[subset].select(range(size),seed=seed)
            return ds
        else:
            raise ValueError("Incorrect data source")

    def get_dataset_binary_balanced(self, split: str, seed: int, subset: str, size: int ,label_col: str) -> Tuple[List[str], List[int]]:
        """ Loads a dataset and filters the dataset into two binary classes.

        Args:
            split: Dataset split to load such as 'train', 'eval'
            seed: Random seed for reproduceable shuffling for the dataset
            subset: Specific subset of the dataset to select from
            size: Maximum number of rows to use from the dataset. If None uses maximum number of rows. If too big gets pruned to the biggest possible size.
            label_col: The column name containing the class labels.

            Returns:
                Tuple[List[str], List[int]]:
                    List of text from the 'text' column
                    List of binary labels


        """

        ds = load_dataset(self.dataset_name, split)[subset].shuffle(seed)
        ds0 = ds.filter(lambda x: x[label_col] == 0)
        ds1 = ds.filter(lambda x: x[label_col] == 1)
        k = min(len(ds0), len(ds1), size // 2 if size else len(ds))
        balanced = concatenate_datasets([ds0.select(range(k)), ds1.select(range(k))]).shuffle(seed)
        return balanced["text"], balanced["label"]