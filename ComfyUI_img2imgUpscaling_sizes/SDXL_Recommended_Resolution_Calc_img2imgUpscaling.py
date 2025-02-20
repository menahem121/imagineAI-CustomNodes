import math

#########################
# BEGIN ACCEPTED RATIOS #
#########################

accepted_ratios_horizontal = {
    "7:4": (1344, 768, 1.750000000),
    "9:7": (1152, 896, 1.285714286),
    "19:13": (1216, 832, 1.461538462),
    "1:2": (704, 1408, 0.500000000),
    "3:1": (1728, 576, 3.000000000),
    "4:1": (2048, 512, 4.000000000),
    "4:3": (1152, 864, 1.333333333),
    "3:2": (1248, 832, 1.500000000),
    "5:2": (1600, 640, 2.500000000),
    "5:3": (1280, 768, 1.666666667),
    "16:9": (1344, 768, 1.750000000),
    "19:7": (1664, 576, 2.888888889),
    "12:5": (1536, 640, 2.400000000),
    "26:7": (1920, 512, 3.750000000),
    "32:9": (1792, 512, 3.500000000),
}

accepted_ratios_vertical = {
    "4:7": (768, 1344, 0.571428571),
    "7:9": (896, 1152, 0.777777778),
    "13:19": (832, 1216, 0.684210526),
    "2:1": (1408, 704, 2.000000000),
    "1:3": (576, 1728, 0.333333333),
    "1:4": (512, 2048, 0.250000000),
    "3:4": (864, 1152, 0.750000000),
    "2:3": (832, 1248, 0.666666667),
    "2:5": (640, 1600, 0.400000000),
    "3:5": (768, 1280, 0.600000000),
    "9:16": (768, 1344, 0.571428571),
    "7:19": (576, 1664, 0.346153846),
    "5:12": (640, 1536, 0.416666667),
    "7:26": (512, 1920, 0.266666667),
    "9:32": (576, 1792, 0.321428571),
}

accepted_ratios_square = {
    "1:1": (1024, 1024, 1.00000000)
}

#########################
# END ACCEPTED RATIOS   #
#########################


##########################################################
# NEW NODE: SDXL Recommended Resolution Calc (Upscaling) #
##########################################################

class SDXL_Recommended_Resolution_Calc_img2imgUpscaling_JPS:
    """
    This node determines the recommended resolution based on the accepted ratio
    while ensuring that images from later upscale passes do not exceed the maximum
    size obtained in a single 1.5× upscale.

    For a given accepted (base) resolution:
      - The fixed base is returned if the input image’s dimensions are below the threshold.
      - If both input dimensions are at least 1.5× the base (i.e. the image comes from a prior upscale),
        then the output is clamped to the maximum allowed resolution.

    Examples (for a 1:1 image):
      Base resolution: 1024×1024
      Maximum allowed (1.5× upscale): 1536×1536
        • Input 1024×1017  → Output 1024×1024
        • Input 1536×1536  → Output 1536×1536
        • Input 1537×1537  → Output 1536×1536
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 8192,
                    "step": 2
                }),
                "target_height": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 8192,
                    "step": 2
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("SDXL_width", "SDXL_height",)

    FUNCTION = "calcSDXLres_img2imgUpscaling"
    CATEGORY = "JPS Nodes/Math"

    def calcSDXLres_img2imgUpscaling(self, target_width, target_height):
        # Avoid division by zero
        if target_height == 0:
            target_height = 1
        target_ratio = target_width / target_height

        # Step 1: Find the closest accepted ratio
        closest_ratio = None
        closest_diff = float('inf')
        for ratio_name, (base_w, base_h, numeric_ratio) in accepted_ratios_horizontal.items():
            diff = abs(numeric_ratio - target_ratio)
            if diff < closest_diff:
                closest_diff = diff
                closest_ratio = ratio_name
        for ratio_name, (base_w, base_h, numeric_ratio) in accepted_ratios_vertical.items():
            diff = abs(numeric_ratio - target_ratio)
            if diff < closest_diff:
                closest_diff = diff
                closest_ratio = ratio_name
        # Also check the square ratio
        sq_w, sq_h, sq_ratio = accepted_ratios_square["1:1"]
        diff = abs(sq_ratio - target_ratio)
        if diff < closest_diff:
            closest_ratio = "1:1"

        # Step 2: Retrieve the fixed base resolution for that ratio
        if closest_ratio in accepted_ratios_horizontal:
            base_width, base_height, _ = accepted_ratios_horizontal[closest_ratio]
        elif closest_ratio in accepted_ratios_vertical:
            base_width, base_height, _ = accepted_ratios_vertical[closest_ratio]
        else:
            base_width, base_height, _ = accepted_ratios_square[closest_ratio]

        # Define the maximum allowed resolution as the first-pass upscale
        upscale_factor = 1.5
        max_width = base_width * upscale_factor
        max_height = base_height * upscale_factor

        # Step 3: Determine the output based on the input dimensions:
        # If the input is at least as large as the max resolution, output the max.
        # Otherwise, output the fixed base resolution.
        if target_width >= max_width and target_height >= max_height:
            recommended_width = int(round(max_width))
            recommended_height = int(round(max_height))
        else:
            recommended_width = base_width
            recommended_height = base_height

        return (recommended_width, recommended_height)


#####################################
# OPTIONAL NODE CLASS MAPPING HOOK  #
#####################################

NODE_CLASS_MAPPINGS = {
    "SDXL Recommended Resolution Calc img2imgUpscaling (JPS)": SDXL_Recommended_Resolution_Calc_img2imgUpscaling_JPS
}
