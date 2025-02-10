from enum import Enum

class INPUT(Enum):
    def IMAGE():
        return ("IMAGE",)
    def MASK():
        return ("MASK",)
    def BOOLEAN(default=True):
        return ("BOOLEAN", dict(default=default))
