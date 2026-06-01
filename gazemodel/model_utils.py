"""
Model instantiation and image preprocessing utilities.
"""

import importlib
import torch
from torchvision import transforms


# ============================================================
#  Dynamic model loading  (from utils/util.py)
# ============================================================

def instantiate_from_cfg(config):
    """Instantiate a class from a config dict with 'type' and 'params' keys."""
    if "type" not in config:
        raise KeyError("Expected key `type` to instantiate.")
    return get_obj_from_str(config["type"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """Dynamically import and return a class from a dotted path string."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# ============================================================
#  Image transforms  (from datasets/helper/image_transform.py)
# ============================================================

def wrap_transforms(image_transforms_type, image_size):
    """Get torchvision transform pipeline. Supports 'basic_imagenet'."""
    if image_transforms_type == 'basic_imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        raise NotImplementedError
