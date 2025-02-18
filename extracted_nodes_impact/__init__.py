from .nodes.pipe_nodes import ToBasicPipe, MaskDetailerPipe
from .nodes.util_nodes import ImageBatchToImageList

NODE_CLASS_MAPPINGS = {
    "ToBasicPipe": ToBasicPipe,
    "MaskDetailerPipe": MaskDetailerPipe,
    "ImageBatchToImageList": ImageBatchToImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ToBasicPipe": "To Basic Pipe",
    "MaskDetailerPipe": "Mask Detailer Pipe",
    "ImageBatchToImageList": "Image Batch To Image List"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
