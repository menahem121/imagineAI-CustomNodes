import numpy as np
import cv2

class DrawMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mask": ("IMAGE", {}),
                "x": ("INT", {"default": 0, "min": 0}),
                "y": ("INT", {"default": 0, "min": 0}),
                "h": ("INT", {"default": 100, "min": 1}),  # mask height
                "w": ("INT", {"default": 100, "min": 1}),  # mask width
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "draw_mask"
    CATEGORY = "examples"

    def create_rounded_mask(self, width, height, radius):
        """
        Creates a binary mask of size (height, width) with rounded corners.
        Corners are rounded using circles with the given radius.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        # Fill central regions
        cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
        cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
        # Draw circles at the corners
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        cv2.circle(mask, (width - radius - 1, radius), radius, 255, -1)
        cv2.circle(mask, (radius, height - radius - 1), radius, 255, -1)
        cv2.circle(mask, (width - radius - 1, height - radius - 1), radius, 255, -1)
        return mask

    def draw_mask(self, image, mask, x, y, h, w):
        # Rename input dimensions for clarity
        mask_height = h
        mask_width = w

        # Convert input images from ComfyUI format (float32 [0,1]) to uint8 [0,255]
        image = (image * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)

        # Resize mask to the specified dimensions (mask_width x mask_height)
        mask_resized = cv2.resize(mask, (mask_width, mask_height))

        # Ensure mask is grayscale (single channel)
        if mask_resized.ndim == 3:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)

        # Convert mask to binary using thresholding
        _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        # Create a rounded mask (using 20% of the smaller mask dimension as radius)
        radius = int(min(mask_width, mask_height) * 0.2)
        rounded_mask = self.create_rounded_mask(mask_width, mask_height, radius)

        # Combine the binary mask with the rounded mask
        mask_final = cv2.bitwise_and(mask_binary, rounded_mask)

        # Prepare an overlay with the same shape as the image
        overlay = np.zeros_like(image)
        # Convert to BGR if image has 3 channels
        if len(overlay.shape) == 3 and overlay.shape[2] == 3:
            mask_colored = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)
        else:
            mask_colored = mask_final

        # Get image dimensions
        img_h, img_w = image.shape[:2]
        # Calculate end positions, ensuring the mask doesn't exceed image bounds
        x_end = min(x + mask_width, img_w)
        y_end = min(y + mask_height, img_h)

        # Place the mask on the overlay (clip if necessary)
        overlay[y:y_end, x:x_end] = mask_colored[0:(y_end - y), 0:(x_end - x)]

        # Blend the overlay with the original image
        blended = cv2.addWeighted(image, 1.0, overlay, 0.5, 0)

        # Convert back to float32 [0,1] for ComfyUI
        image_out = blended.astype(np.float32) / 255.0
        return (image_out,)

# Node registration dictionaries
NODE_CLASS_MAPPINGS = { "draw_mask_node": DrawMaskNode }
NODE_DISPLAY_NAME_MAPPINGS = { "draw_mask_node": "Draw Mask on Image 121" }
