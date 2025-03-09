import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw

def to_numpy(x):
    """Convert various input types to numpy arrays"""
    if isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "cpu"):
        return x.cpu().numpy()
    else:
        return np.array(x)

class DrawMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),   # Reference image (canvas)
                "mask": ("MASK", {}),     # Base mask to overlay (may have a different size)
                "x": ("INT", {"default": 0, "min": 0}),
                "y": ("INT", {"default": 0, "min": 0}),
                "w": ("INT", {"default": 100, "min": 1}),  # Desired rectangle width
                "h": ("INT", {"default": 100, "min": 1}),  # Desired rectangle height
                "radius": ("INT", {"default": 20, "min": 0}),  # Rounded corner radius
            },
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask_out",)
    FUNCTION = "make_mask"
    CATEGORY = "mask"
    
    def make_mask(self, image, mask, x, y, w, h, radius):
        # Get canvas dimensions from the reference image
        # ComfyUI images are tensors; assume image shape is either [H,W,C] or [B,H,W,C]
        image_np = to_numpy(image)
        if image_np.ndim == 4:
            # Assume batch dimension exists; take the first image
            canvas_h, canvas_w = image_np[0].shape[:2]
        else:
            canvas_h, canvas_w = image_np.shape[:2]
        
        # Ensure the input mask is a torch tensor with shape [B, H_in, W_in]
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(to_numpy(mask))
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        # Resize the input mask to match the canvas dimensions if needed
        if mask.shape[-2:] != (canvas_h, canvas_w):
            mask = F.interpolate(mask.unsqueeze(0), size=(canvas_h, canvas_w), mode='nearest')
            mask = mask.squeeze(0)
        
        # Create a new mask canvas (using PIL) of the same dimensions as the reference image
        canvas = Image.new("L", (canvas_w, canvas_h), 0)  # "L" mode: 8-bit grayscale
        draw = ImageDraw.Draw(canvas)
        
        # Draw a rounded rectangle at (x, y) with size (w, h)
        rect_coords = [x, y, x + w, y + h]
        draw.rounded_rectangle(rect_coords, radius=radius, fill=255)
        
        # Convert the PIL image to a numpy array (float32 normalized to [0,1])
        new_mask_np = np.array(canvas, dtype=np.float32) / 255.0
        
        # Convert to torch tensor and add batch dimension if necessary
        new_mask = torch.from_numpy(new_mask_np)
        if new_mask.ndim == 2:
            new_mask = new_mask.unsqueeze(0)
        
        # Move the new mask to the same device as the input mask
        new_mask = new_mask.to(mask.device)
        
        # Combine the input mask (resized to canvas size) and the new mask via pixel-wise maximum
        # This overlays the drawn rounded rectangle onto the base mask
        combined_mask = torch.maximum(mask, new_mask)
        
        return (combined_mask,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "draw_mask_node121": DrawMaskNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "draw_mask_node121": "Draw Mask"
}