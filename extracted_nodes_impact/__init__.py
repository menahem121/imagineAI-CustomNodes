from .nodes.util_nodes import ImageBatchToImageList

NODE_CLASS_MAPPINGS = {
    "ImageBatchToImageList": ImageBatchToImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchToImageList": "Image Batch To Image List"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']