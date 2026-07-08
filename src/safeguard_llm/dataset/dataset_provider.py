from abc import ABC

class DatasetProvider(ABC): 
    def __init__(self, dataset_name, source): 
        self.dataset_name = dataset_name
        self.dataset_source = source

    def get_dataset(self):
        if self.source == "hf":
            from datasets import Dataset
                    
