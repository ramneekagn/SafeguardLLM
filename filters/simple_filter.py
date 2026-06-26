from IO_filter import IOFilter  

class SimpleFilter(IOFilter): 
    def __init__(self):
        super().__init__()

    def approve(input: str) -> bool: 
        if len(input) > 30: 
            return True
        return False
    