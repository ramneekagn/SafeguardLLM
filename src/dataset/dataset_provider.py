from typing import Tuple, Any, List
class DatasetProvider(): 
    def __init__(self, dataset_name, source): 
        self.dataset_name = dataset_name
        self.dataset_source = source

    def get_dataset(self, split,seed,size) -> Tuple[List[str], List[int]]:
        if self.dataset_source == "hf":
            from datasets import load_dataset
            ds = load_dataset(self.dataset_name,split)            
            ds = ds.shuffle(seed)
            print(ds)
            ds = ds["train"].select(range(size))
            return ds
        else:
            raise ValueError("Incorrect data source")
        