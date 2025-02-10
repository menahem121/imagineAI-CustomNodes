# ComfyUI Inpaint Preprocessor Node

This is a standalone version of the Inpaint Preprocessor node from the ComfyUI ControlNet Auxiliary Preprocessors collection. It provides the exact same functionality as the original node.

## Functionality

The Inpaint Preprocessor node takes an image and a mask as input, and prepares them for inpainting by:
1. Resizing the mask to match the image dimensions using bilinear interpolation
2. Applying the mask to the image by setting masked pixels to either -1.0 (default) or 0.0 (when black_pixel_for_xinsir_cn is enabled)

### Inputs
- `image`: The input image (required)
- `mask`: The input mask (required)
- `black_pixel_for_xinsir_cn`: Optional boolean parameter to use 0.0 instead of -1.0 for masked pixels (default: False)

### Output
- Preprocessed image ready for inpainting

## Installation

1. Copy the `comfyui_inpaint_preprocessor` directory to your ComfyUI's `custom_nodes` directory
2. Restart ComfyUI
3. The node will be available in the node menu under "ControlNet Preprocessors/others" as "Inpaint Preprocessor"

## Dependencies

This node has no additional dependencies beyond PyTorch, which is already provided by ComfyUI.

## Note

This is a minimal, standalone version that provides the exact same functionality as the original Inpaint Preprocessor node. The code has been extracted without modification to ensure identical behavior.
