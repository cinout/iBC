import random
import numpy as np
from PIL import Image
import torch


def tensor_back_to_PIL(input):
    input = torch.permute(input, (1, 2, 0))
    input = input * 255.0
    input = torch.clamp(input, 0, 255)
    input = np.array(input, dtype=np.uint8)
    input = Image.fromarray(input)

    return input


class PoisonHTBA:
    def __init__(self, args):
        self.args = args
        self.alpha = 0.0
        self.watermark = "triggers/HTBA_trigger_10.png"
        self.watermark_width = int(50.0 / 224 * args.image_size)
        self.location_min = 0.2
        self.location_max = 0.8

    def Poison_HTBA(self, x_train, y_train):
        # x_train shape: torch tensors [50000, 3, 32, 32]; value range: [0, 1]
        if x_train.shape[0] == 0:
            return x_train

        img_watermark = Image.open(self.watermark).convert("RGBA")
        w_width, w_height = self.watermark_width, int(
            img_watermark.size[1] * self.watermark_width / img_watermark.size[0]
        )
        img_watermark = img_watermark.resize((w_width, w_height))

        poisoned_images = []
        for i, base_image in enumerate(x_train):

            base_image = tensor_back_to_PIL(base_image)
            base_image = base_image.convert("RGBA")
            width, height = base_image.size
            transparent = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            loc_min_w = int(width * self.location_min)
            loc_max_w = int(width * self.location_max - w_width)
            if loc_max_w < loc_min_w:
                loc_max_w = loc_min_w

            loc_min_h = int(height * self.location_min)
            loc_max_h = int(height * self.location_max - w_height)
            if loc_max_h < loc_min_h:
                loc_max_h = loc_min_h
            location = (
                random.randint(loc_min_w, loc_max_w),
                random.randint(loc_min_h, loc_max_h),
            )

            transparent.paste(img_watermark, location)
            na = np.array(transparent).astype(float)
            transparent = Image.fromarray(na.astype(np.uint8))
            na = np.array(base_image).astype(float)
            na[..., 3][
                location[1] : (location[1] + w_height),
                location[0] : (location[0] + w_width),
            ] *= self.alpha
            base_image = Image.fromarray(na.astype(np.uint8))
            transparent = Image.alpha_composite(transparent, base_image)
            transparent = transparent.convert("RGB")

            # CONVERT to the tensor format
            transparent = np.asarray(transparent).astype(np.float32) / 255.0
            transparent = torch.tensor(transparent)
            transparent = torch.permute(
                transparent, (2, 0, 1)
            )  # shape: [c=3, h, w], value: [0, 1]
            poisoned_images.append(transparent)

        poisoned_images = torch.stack(poisoned_images, dim=0)

        return poisoned_images, y_train
