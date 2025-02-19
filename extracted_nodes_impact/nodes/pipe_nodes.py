import torch
import comfy.samplers
import numpy as np
import nodes
import comfy.utils
import inspect

def make_2d_mask(mask):
    if len(mask.shape) == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        mask = mask.squeeze(0)
    return mask

def tensor_gaussian_blur_mask(tensor, kernel_size):
    """Apply Gaussian blur to a tensor mask"""
    if kernel_size <= 0:
        return tensor
    
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    padding = int((kernel_size - 1) / 2)
    gaussian_blur = torch.nn.GaussianBlur(kernel_size, sigma=1.0, padding=padding)
    blurred = gaussian_blur(tensor)
    return blurred.squeeze(0).squeeze(0)

def tensor_paste(destination, source, left_top_offset, mask=None):
    """Paste tensor with optional mask"""
    x, y = left_top_offset
    source_h, source_w = source.shape[2:]
    
    if mask is not None:
        mask = mask.unsqueeze(0).unsqueeze(0)
        destination[:, :, y:y+source_h, x:x+source_w] = (
            destination[:, :, y:y+source_h, x:x+source_w] * (1 - mask) +
            source * mask
        )
    else:
        destination[:, :, y:y+source_h, x:x+source_w] = source

def mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size=1, label='A', crop_min_size=None, detailer_hook=None, is_contour=True):
    drop_size = max(drop_size, 1)
    if mask is None:
        print("[mask_to_segs] Cannot operate: MASK is empty.")
        return ([],)

    if isinstance(mask, np.ndarray):
        pass  # `mask` is already a NumPy array
    else:
        try:
            mask = mask.numpy()
        except AttributeError:
            print("[mask_to_segs] Cannot operate: MASK is not a NumPy array or Tensor.")
            return ([],)

    result = []

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)

    for i in range(mask.shape[0]):
        mask_i = mask[i]

        if combined:
            indices = np.nonzero(mask_i)
            if len(indices[0]) > 0 and len(indices[1]) > 0:
                bbox = (
                    np.min(indices[1]),
                    np.min(indices[0]),
                    np.max(indices[1]),
                    np.max(indices[0]),
                )
                crop_region = make_crop_region(
                    mask_i.shape[1], mask_i.shape[0], bbox, crop_factor
                )
                x1, y1, x2, y2 = crop_region

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(mask_i.shape[1], mask_i.shape[0], bbox, crop_region)

                if x2 - x1 > 0 and y2 - y1 > 0:
                    cropped_mask = mask_i[y1:y2, x1:x2]

                    if bbox_fill:
                        bx1, by1, bx2, by2 = bbox
                        cropped_mask = cropped_mask.copy()
                        cropped_mask[by1:by2, bx1:bx2] = 1.0

                    if cropped_mask is not None:
                        item = SEG(None, cropped_mask, 1.0, crop_region, bbox, label, None)
                        result.append(item)

    return (mask.shape[1], mask.shape[2]), result

def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        if crop_w < crop_min_size:
            crop_w = crop_min_size

        if crop_h < crop_min_size:
            crop_h = crop_min_size

    x1_crop = int(x1 + bbox_w/2 - crop_w/2)
    y1_crop = int(y1 + bbox_h/2 - crop_h/2)

    x2_crop = x1_crop + int(crop_w)
    y2_crop = y1_crop + int(crop_h)

    # make sure crop is within image
    if x1_crop < 0:
        x2_crop -= x1_crop
        x1_crop = 0
    if y1_crop < 0:
        y2_crop -= y1_crop
        y1_crop = 0
    if x2_crop > w:
        x1_crop -= (x2_crop - w)
        x2_crop = w
    if y2_crop > h:
        y1_crop -= (y2_crop - h)
        y2_crop = h

    return x1_crop, y1_crop, x2_crop, y2_crop

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper

class ToBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "model": ("MODEL",),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     },
                }

    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("basic_pipe", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Pipe"

    def doit(self, model, clip, vae, positive, negative):
        pipe = (model, clip, vae, positive, negative)
        return (pipe, )

class MaskDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "mask": ("MASK", ),
                    "basic_pipe": ("BASIC_PIPE",),
                    "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                    "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "mask bbox", "label_off": "crop region"}),
                    "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                    "mask_mode": ("BOOLEAN", {"default": True, "label_on": "masked only", "label_off": "whole"}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                    "drop_size": ("INT", {"min": 1, "max": 2048, "step": 1, "default": 10}),
                    "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                    "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                   },
                "optional": {
                    "refiner_basic_pipe_opt": ("BASIC_PIPE", ),
                    "detailer_hook": ("DETAILER_HOOK",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "contour_fill": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                   }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "BASIC_PIPE", "BASIC_PIPE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "basic_pipe", "refiner_basic_pipe_opt")
    OUTPUT_IS_LIST = (False, True, True, False, False)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    def doit(self, image, mask, basic_pipe, guide_size, guide_size_for, max_size, mask_mode,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             feather, crop_factor, drop_size, refiner_ratio, batch_size, cycle=1,
             refiner_basic_pipe_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=0,
             bbox_fill=False, contour_fill=True, scheduler_func_opt=None):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: MaskDetailer does not allow image batches.')

        model, clip, vae, positive, negative = basic_pipe

        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        # create segs
        if mask is not None:
            mask = make_2d_mask(mask)
            segs = mask_to_segs(mask, False, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)
        else:
            segs = ((image.shape[1], image.shape[2]), [])

        enhanced_img_batch = None
        cropped_enhanced_list = []
        cropped_enhanced_alpha_list = []

        for i in range(batch_size):
            if mask is not None:
                # Process each segment
                for seg in segs[1]:
                    # Extract the cropped region
                    x1, y1, x2, y2 = seg.crop_region
                    cropped_image = image[:, :, y1:y2, x1:x2]
                    cropped_mask = torch.from_numpy(seg.cropped_mask).unsqueeze(0).unsqueeze(0)

                    # Create latent from cropped image
                    if inpaint_model:
                        # Use inpaint model conditioning
                        imc_encode = nodes.InpaintModelConditioning().encode
                        if 'noise_mask' in inspect.signature(imc_encode).parameters:
                            positive, negative, latent = imc_encode(positive, negative, cropped_image, vae, mask=cropped_mask, noise_mask=True)
                        else:
                            positive, negative, latent = imc_encode(positive, negative, cropped_image, vae, cropped_mask)
                    else:
                        latent = nodes.VAEEncode().encode(vae, cropped_image)[0]
                        if mask_mode:
                            latent['noise_mask'] = cropped_mask

                    # Sample
                    sampled = nodes.KSampler().sample(model, seed+i, steps, cfg, sampler_name, scheduler,
                                                     positive, negative, latent, denoise)[0]

                    # Decode
                    enhanced = nodes.VAEDecode().decode(vae, sampled)[0]

                    # Composite back
                    if mask_mode:
                        # Feather the mask
                        if feather > 0:
                            cropped_mask = tensor_gaussian_blur_mask(cropped_mask, feather)
                        # Composite
                        enhanced = enhanced * cropped_mask + cropped_image * (1 - cropped_mask)

                    # Add to batch
                    if enhanced_img_batch is None:
                        enhanced_img_batch = image.clone()
                    tensor_paste(enhanced_img_batch, enhanced, (x1, y1), cropped_mask)

                    cropped_enhanced_list.append(enhanced)
                    cropped_enhanced_alpha_list.append(enhanced * cropped_mask)

            else:
                enhanced_img_batch = image

        # Set fallback images if needed
        if len(cropped_enhanced_list) == 0:
            empty = torch.zeros((1, 3, 64, 64))
            cropped_enhanced_list = [empty]
            cropped_enhanced_alpha_list = [empty]

        return enhanced_img_batch, cropped_enhanced_list, cropped_enhanced_alpha_list, basic_pipe, refiner_basic_pipe_opt
