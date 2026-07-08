from typing import Tuple, Any, List
from datasets import load_dataset, concatenate_datasets

class DatasetProvider(): 
    def __init__(self, dataset_name, source): 
        self.dataset_name = dataset_name
        self.dataset_source = source

    def get_dataset(self, split,seed,subset, size) -> Tuple[List[str], List[int]]:
        if self.dataset_source == "hf":
            from datasets import load_dataset
            ds = load_dataset(self.dataset_name,split)            
            ds = ds.shuffle(seed)
            ds = ds[subset].select(range(size))
            return ds
        else:
            raise ValueError("Incorrect data source")
    def get_dataset_binary_balanced(self, split, seed, subset, size,label_col) -> Tuple[List[str], List[int]]:
        ds = load_dataset(self.dataset_name, split)[subset].shuffle(seed)
        ds0 = ds.filter(lambda x: x[label_col] == 0)
        ds1 = ds.filter(lambda x: x[label_col] == 1)
        k = min(len(ds0), len(ds1), size // 2 if size else len(ds))
        balanced = concatenate_datasets([ds0.select(range(k)), ds1.select(range(k))]).shuffle(seed)
        return balanced["text"], balanced["label"]