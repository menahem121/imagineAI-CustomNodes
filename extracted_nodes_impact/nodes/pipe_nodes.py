import torch
import comfy.samplers

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

        # Create empty tensors for fallback
        empty_pil = torch.zeros((1, 64, 64, 3))
        empty_mask = torch.zeros((64, 64))

        return image, [empty_pil], [empty_pil], basic_pipe, refiner_basic_pipe_opt
