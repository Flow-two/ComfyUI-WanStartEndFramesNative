from .nodes.nodes import *

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "WanImageToVideo_F2":           WanImageToVideo_F2,
    "GetImagesFromBatchRanged_F2":  GetImagesFromBatchRanged_F2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanImageToVideo_F2":           "WanImageToVideo (Flow2)",
    "GetImagesFromBatchRanged_F2":  "Get Images From Batch Ranged"
}