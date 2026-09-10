"""
Uncomment this file if needed later
"""

import glob
from typing import Tuple
import torch
import torch.nn as nn
import albumentations
from scipy.fftpack import dct
import numpy as np
import cv2
import math, random
import imgaug.augmenters as iaa
from skimage.transform import rotate as im_rotate
from kornia.filters import gaussian_blur2d


def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def addnoise(img, complex_gaussian):
    if complex_gaussian:
        aug = albumentations.GaussNoise(
            p=1,
            mean=random.randrange(-10, 30, 1),
            var_limit=(10, 70),
            noise_scale_factor=random.uniform(0.5, 1),
            # noise_scale_factor=random.uniform(0.25, 1),
            # per_channel=random.choice([True, False]),
        )
    else:
        aug = albumentations.GaussNoise(p=1, mean=25, var_limit=(10, 70))

    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def randshadow(img, image_size):
    aug = albumentations.RandomShadow(
        p=1,
        shadow_roi=(0, 0.5, 1, 1),
        num_shadows_limit=(1, 1),
        shadow_dimension=random.randrange(3, 5),
        shadow_intensity_range=(0.6, 0.75),
    )
    test = (img * 255).astype(np.uint8)
    augmented = aug(image=cv2.resize(test, (image_size, image_size)))
    auged = augmented["image"] / 255
    return auged


def rand_rain(img, image_size):
    aug = albumentations.RandomRain(
        p=1,
        drop_length=(
            random.randrange(3, 8) if image_size == 64 else random.randrange(1, 3)
        ),
        drop_width=random.randrange(1, 2) if image_size == 64 else 1,
        drop_color=(
            random.randrange(0, 255),
            random.randrange(0, 255),
            random.randrange(0, 255),
        ),
        blur_value=1,
        brightness_coefficient=0.9,
        rain_type="drizzle",
    )
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def rand_sunflare(img, image_size):
    aug = albumentations.RandomSunFlare(
        p=1,
        src_radius=int(image_size * 0.2),
        src_color=(
            random.randrange(0, 255),
            random.randrange(0, 255),
            random.randrange(0, 255),
        ),
        num_flare_circles_range=(2, 3),
        angle_range=(0, 1),
    )
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def posterize(img):
    aug = albumentations.Posterize(p=1, num_bits=random.randrange(2, 3))
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def pixel_dropout(img):
    aug = albumentations.PixelDropout(p=1)
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def spatter_rain(img):
    aug = albumentations.Spatter(p=1, mode="rain")
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def spatter_mud(img):
    aug = albumentations.Spatter(p=1, mode="mud")
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def defocus(img):
    aug = albumentations.Defocus(p=1, radius=3)
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


class CutPasteNormal(object):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.01, 0.05], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        # img.shape: [32, 32, 3]; value range: [0, 1]

        h = img.shape[0]
        w = img.shape[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        patch = img[
            from_location_h : from_location_h + cut_h,
            from_location_w : from_location_w + cut_w,
            :,
        ]

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        img[
            to_location_h : to_location_h + cut_h,
            to_location_w : to_location_w + cut_w,
            :,
        ] = patch

        return img


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(
        np.repeat(
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0
        ),
        d[1],
        axis=1,
    )
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]
    )


augmenters = [
    iaa.GammaContrast((0.5, 2.0), per_channel=True),
    iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
    iaa.pillike.EnhanceSharpness(),
    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
    iaa.Solarize(0.5, threshold=(32, 128)),
    iaa.Posterize(),
    iaa.Invert(),
    iaa.pillike.Autocontrast(),
    iaa.pillike.Equalize(),
    iaa.Affine(rotate=(-45, 45)),
]

anomaly_source_paths = sorted(glob.glob("./datasets/dtd/images/*/*.jpg"))


def draem_augment(image, image_size):
    aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential(
        [
            augmenters[aug_ind[0]],
            augmenters[aug_ind[1]],
            augmenters[aug_ind[2]],
        ]
    )
    perlin_scale = 6
    min_perlin_scale = 0

    anomaly_source_idx = torch.randint(0, len(anomaly_source_paths), (1,)).item()
    anomaly_source_path = anomaly_source_paths[anomaly_source_idx]
    anomaly_source_img = cv2.imread(anomaly_source_path)
    anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(image_size, image_size))
    anomaly_img_augmented = aug(image=anomaly_source_img)

    perlin_scalex = 2 ** (
        torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
    )
    perlin_scaley = 2 ** (
        torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
    )
    perlin_noise = rand_perlin_2d_np(
        (image_size, image_size), (perlin_scalex, perlin_scaley)
    )

    perlin_noise = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise)

    threshold = 0.5
    perlin_thr = np.where(
        perlin_noise > threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise),
    )
    perlin_thr = np.expand_dims(perlin_thr, axis=2)
    img_thr = (
        anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
    )  # range [0, 1], shape: [32, 32, 3]
    beta = torch.rand(1).numpy()[0] * 0.8

    augmented_image = (
        image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
    )
    augmented_image = augmented_image.astype(np.float32)
    msk = (perlin_thr).astype(np.float32)
    augmented_image = msk * augmented_image + (1 - msk) * image

    return augmented_image  # shape: [32, 32, 3], values in [0, 1], numpy


def ceil(x: float):
    return int(np.ceil(x))


def floor(x: float):
    return int(np.floor(x))


def confetti_noise(
    size: torch.Size,
    p: float = 0.01,
    blobshaperange: Tuple[Tuple[int, int], Tuple[int, int]] = ((3, 3), (5, 5)),
    fillval: int = 255,
    backval: int = 0,
    ensureblob: bool = True,
    awgn: float = 0.0,
    clamp: bool = False,
    onlysquared: bool = True,
    rotation: int = 0,
    colorrange: Tuple[int, int] = None,
) -> torch.Tensor:
    """
    Generates "confetti" noise, as seen in the paper.
    The noise is based on sampling randomly many rectangles (in the following called blobs) at random positions.
    Additionally, all blobs are of random size (within some range), of random rotation, and of random color.
    The color is randomly chosen per blob, thus consistent within one blob.

    :param size: size of the overall noise image(s), should be (n x h x w) or (n x c x h x w), i.e.
        number of samples, channels, height, width. Blobs are grayscaled for (n x h x w) or c == 1.
    :param p: the probability of inserting a blob per pixel.
        The average number of blobs in the image is p * h * w.

    :param blobshaperange: limits the random size of the blobs. For ((h0, h1), (w0, w1)), all blobs' width
        is ensured to be in {w0, ..., w1}, and height to be in {h0, ..., h1}.
    :param fillval: if the color is not randomly chosen (see colored parameter), this sets the color of all blobs.
        This is also the maximum value used for clamping (see clamp parameter). Can be negative.
    :param backval: the background pixel value, i.e. the color of pixels in the noise image that are not part
         of a blob. Also used for clamping.
    :param ensureblob: whether to ensure that there is at least one blob per noise image.
    :param awgn: amount of additive white gaussian noise added to all blobs.
    :param clamp: whether to clamp all noise image to the pixel value range (backval, fillval).
    :param onlysquared: whether to restrict the blobs to be squares only.
    :param rotation: the maximum amount of rotation (in degrees)
    :param colorrange: the range of possible color values for each blob and channel.
        Defaults to None, where the blobs are not colored, but instead parameter fillval is used.
        First value can be negative.

    :return: torch tensor containing n noise images. Either (n x c x h x w) or (n x h x w), depending on size.
    """
    assert len(size) == 4 or len(size) == 3, "size must be n x c x h x w"
    if isinstance(blobshaperange[0], int) and isinstance(blobshaperange[1], int):
        blobshaperange = (blobshaperange, blobshaperange)
    assert len(blobshaperange) == 2
    assert len(blobshaperange[0]) == 2 and len(blobshaperange[1]) == 2
    assert colorrange is None or len(size) == 4 and size[1] == 3
    out_size = size
    colors = []
    if len(size) == 3:
        size = (size[0], 1, size[1], size[2])  # add channel dimension
    else:
        size = tuple(
            size
        )  # Tensor(torch.size) -> tensor of shape size, Tensor((x, y)) -> Tensor with 2 elements x & y
    mask = (torch.rand((size[0], size[2], size[3])) < p).unsqueeze(
        1
    )  # mask[i, j, k] == 1 for center of blob
    while ensureblob and (mask.view(mask.size(0), -1).sum(1).min() == 0):
        idx = (mask.view(mask.size(0), -1).sum(1) == 0).nonzero().squeeze()
        s = idx.size(0) if len(idx.shape) > 0 else 1
        mask[idx] = torch.rand((s, 1, size[2], size[3])) < p
    res = torch.empty(size).fill_(backval).int()
    idx = mask.nonzero()  # [(idn, idz, idy, idx), ...] = indices of blob centers
    if idx.reshape(-1).size(0) == 0:
        return torch.zeros(out_size).int()

    all_shps = [
        (x, y)
        for x in range(blobshaperange[0][0], blobshaperange[1][0] + 1)
        for y in range(blobshaperange[0][1], blobshaperange[1][1] + 1)
        if not onlysquared or x == y
    ]
    picks = (
        torch.FloatTensor(idx.size(0)).uniform_(0, len(all_shps)).int()
    )  # for each blob center pick a shape
    nidx = []
    for n, blobshape in enumerate(all_shps):
        if (picks == n).sum() < 1:
            continue
        bhs = range(
            -(blobshape[0] // 2) if blobshape[0] % 2 != 0 else -(blobshape[0] // 2) + 1,
            blobshape[0] // 2 + 1,
        )
        bws = range(
            -(blobshape[1] // 2) if blobshape[1] % 2 != 0 else -(blobshape[1] // 2) + 1,
            blobshape[1] // 2 + 1,
        )
        extends = torch.stack(
            [
                torch.zeros(len(bhs) * len(bws)).long(),
                torch.zeros(len(bhs) * len(bws)).long(),
                torch.arange(bhs.start, bhs.stop).repeat(len(bws)),
                torch.arange(bws.start, bws.stop)
                .unsqueeze(1)
                .repeat(1, len(bhs))
                .reshape(-1),
            ]
        ).transpose(0, 1)
        nid = idx[picks == n].unsqueeze(1) + extends.unsqueeze(0)
        if colorrange is not None:
            col = (
                torch.randint(colorrange[0], colorrange[1], (3,))[:, None]
                .repeat(1, nid.reshape(-1, nid.size(-1)).size(0))
                .int()
            )
            colors.append(col)
        nid = nid.reshape(-1, extends.size(1))
        nid = torch.max(
            torch.min(nid, torch.LongTensor(size) - 1), torch.LongTensor([0, 0, 0, 0])
        )
        nidx.append(nid)
    idx = torch.cat(nidx)  # all pixel indices that blobs cover, not only center indices
    shp = res[idx.transpose(0, 1).numpy()].shape
    if colorrange is not None:
        colors = torch.cat(colors, dim=1)
        gnoise = (torch.randn(3, *shp) * awgn).int() if awgn != 0 else (0, 0, 0)
        res[idx.transpose(0, 1).numpy()] = colors[0] + gnoise[0]
        res[(idx + torch.LongTensor((0, 1, 0, 0))).transpose(0, 1).numpy()] = (
            colors[1] + gnoise[1]
        )
        res[(idx + torch.LongTensor((0, 2, 0, 0))).transpose(0, 1).numpy()] = (
            colors[2] + gnoise[2]
        )
    else:
        gnoise = (torch.randn(shp) * awgn).int() if awgn != 0 else 0
        res[idx.transpose(0, 1).numpy()] = torch.ones(shp).int() * fillval + gnoise
        res = res[:, 0, :, :]
        if len(out_size) == 4:
            res = res.unsqueeze(1).repeat(1, out_size[1], 1, 1)
    if clamp:
        res = (
            res.clamp(backval, fillval)
            if backval < fillval
            else res.clamp(fillval, backval)
        )
    mask = mask[:, 0, :, :]
    if rotation > 0:
        idx = mask.nonzero()
        res = res.unsqueeze(1) if res.dim() != 4 else res
        res = res.transpose(1, 3).transpose(1, 2)
        for pick, blbctr in zip(picks, mask.nonzero()):
            rot = np.random.uniform(-rotation, rotation)
            p1, p2 = all_shps[pick]
            dims = (
                blbctr[0],
                slice(
                    max(blbctr[1] - floor(0.75 * p1), 0),
                    min(blbctr[1] + ceil(0.75 * p1), res.size(1) - 1),
                ),
                slice(
                    max(blbctr[2] - floor(0.75 * p2), 0),
                    min(blbctr[2] + ceil(0.75 * p2), res.size(2) - 1),
                ),
                ...,
            )
            res[dims] = torch.from_numpy(
                im_rotate(
                    res[dims].float(),
                    rot,
                    order=0,
                    cval=0,
                    center=(blbctr[1] - dims[1].start, blbctr[2] - dims[2].start),
                    clip=False,
                )
            ).int()
        res = res.transpose(1, 2).transpose(1, 3)
        res = res.squeeze() if len(out_size) != 4 else res
    return res


def smooth_noise(
    img: torch.Tensor, ksize: int, std: float, p: float = 1.0, inplace: bool = True
) -> torch.Tensor:
    """
    Smoothens (blurs) the given noise images with a Gaussian kernel.
    :param img: torch tensor (n x c x h x w).
    :param ksize: the kernel size used for the Gaussian kernel.
    :param std: the standard deviation used for the Gaussian kernel.
    :param p: the chance smoothen an image, on average smoothens p * n images.
    :param inplace: whether to apply the operation inplace.
    """
    if not inplace:
        img = img.clone()
    ksize = ksize if ksize % 2 == 1 else ksize - 1
    picks = torch.from_numpy(np.random.binomial(1, p, size=img.size(0))).bool()
    if picks.sum() > 0:
        img[picks] = gaussian_blur2d(img[picks].float(), (ksize,) * 2, (std,) * 2).int()
    return img


def malformed_normal(
    generated_noise: torch.Tensor,
    norm: torch.Tensor,
    brightness_threshold: float = 0.11 * 255,
):
    """
    Creates a dataset based on the nominal classes of a given dataset and generated noise anomalies.
    Unlike above, the noise images are not directly utilized as anomalies, but added to nominal samples to
    create malformed normal anomalies.
    :param outlier_classes: a list of all outlier class indices.
    :param generated_noise: torch tensor of noise images (might also be Outlier Exposure based noise) (n x c x h x w).

    :param norm: torch tensor of nominal images (n x c x h x w).

    :param nom_class: the index of the class that is considered nominal.
    :param train_set: some training dataset.
    :param gt: whether to provide ground-truth maps as well.
    :param brightness_threshold: if the average brightness (averaged over color channels) of a pixel exceeds this
        threshold, the noise image's pixel value is subtracted instead of added.
        This avoids adding brightness values to bright pixels, where approximately no effect is achieved at all.
    :return: a modified dataset, with training data consisting of nominal samples and artificial anomalies.
    """
    assert (norm.dim() == 4 or norm.dim() == 3) and generated_noise.shape == norm.shape
    norm_dim = norm.dim()
    if norm_dim == 3:
        norm, generated_noise = norm.unsqueeze(1), generated_noise.unsqueeze(
            1
        )  # assuming ch dim is skipped
    anom = norm.clone()

    # invert noise for bright regions (bright regions are considered being on average > brightness_threshold)
    generated_noise = generated_noise.int()
    bright_regions = norm.sum(1) > brightness_threshold * norm.shape[1]
    for ch in range(norm.shape[1]):
        gnch = generated_noise[:, ch]
        gnch[bright_regions] = gnch[bright_regions] * -1
        generated_noise[:, ch] = gnch

    anom = (anom.int() + generated_noise).clamp(0, 255).byte()

    return anom


def confetti_poisoning(clean_image, image_size):
    # clean_image: shape: [32, 32, 3]; value range: [0, 1], numpy

    generated_noise_rgb = confetti_noise(
        (1, 3, image_size, image_size),
        1.2 / image_size**2,
        ((2, 2), (12, 12)),
        fillval=255,
        clamp=False,
        awgn=0,
        rotation=45,
        colorrange=(-256, 0),
    )
    generated_noise = confetti_noise(
        (1, 3, image_size, image_size),
        1.2 / image_size**2,
        ((2, 2), (12, 12)),
        fillval=-255,
        clamp=False,
        awgn=0,
        rotation=45,
    )
    generated_noise = generated_noise_rgb + generated_noise
    generated_noise = smooth_noise(
        generated_noise, 3, 2, 1.0
    )  # shape: [1, 3, 32, 32], value: [-255,0], torch tensor

    clean_image = torch.tensor(clean_image)
    clean_image = torch.permute(clean_image, (2, 0, 1))
    clean_image = clean_image * 255.0
    clean_image = torch.clamp(clean_image, 0, 255)
    clean_image = torch.unsqueeze(clean_image, 0)  # [1, 3, 32, 32], value 0~255

    poisoned_image = malformed_normal(
        generated_noise, clean_image
    )  # [1, 3, 32, 32], value 0~255

    # uncommet
    poisoned_image = poisoned_image.squeeze(0)
    poisoned_image = torch.permute(
        poisoned_image, (1, 2, 0)
    )  # [32, 32，3], value 0~255
    poisoned_image = (
        np.asarray(poisoned_image).astype(np.float32) / 255.0
    )  # [32, 32, 3]; value range: [0, 1], numpy

    return poisoned_image


def patching_train(
    clean_sample,
    x_train,
    image_size,
    ensemble_id,
    frequency_attack_trigger_ids,
    complex_gaussian,
):
    """
    this code conducts a patching procedure with random white blocks or random noise block
    clean_sample.shape: [32, 32, 3]; value range: [0, 1]
    """
    # attack = np.random.choice([2, 3, 4, 5, 6, 7, 8, 9, 12], 1)[0]
    # attack = np.random.choice([1, 2, 5, 7, 8], 1)[0]
    attack = np.random.choice(frequency_attack_trigger_ids, 1)[0]

    pat_size_x = np.random.randint(2, 8)
    pat_size_y = np.random.randint(2, 8)
    output = np.copy(clean_sample)
    cutpaste_normal = CutPasteNormal()

    if attack == 0:
        # white block
        block = np.ones((pat_size_x, pat_size_y, 3))
    elif attack == 1:
        # random color block
        block = np.random.rand(pat_size_x, pat_size_y, 3)
    elif attack == 2:
        # Gaussian noise
        return addnoise(output, complex_gaussian)
    elif attack == 3:
        # drawing shadows
        return randshadow(output, image_size)
    elif attack == 4:
        # randomly blend with another image
        randind = np.random.randint(x_train.shape[0])
        tri = x_train[randind]
        mid = output + 0.3 * tri
        mid[mid > 1] = 1
        return mid
    elif attack == 5:
        # CUT-PASTE normal
        return cutpaste_normal(output)
    elif attack == 6:
        # DRAEM
        return draem_augment(output, image_size)
    elif attack == 7:
        # Confetti
        return confetti_poisoning(output, image_size)
    elif attack == 8:
        return rand_rain(output, image_size)
    elif attack == 9:
        return rand_sunflare(output, image_size)
    elif attack == 10:
        return posterize(output)
    elif attack == 11:
        return pixel_dropout(output)
    elif attack == 12:
        return spatter_rain(output)
    elif attack == 13:
        return spatter_mud(output)
    elif attack == 14:
        return defocus(output)

    margin = np.random.randint(0, 6)
    rand_loc = np.random.randint(0, 4)

    if rand_loc == 0:
        output[margin : margin + pat_size_x, margin : margin + pat_size_y, :] = (
            block  # upper left
        )
    elif rand_loc == 1:
        output[
            margin : margin + pat_size_x,
            image_size - margin - pat_size_y : image_size - margin,
            :,
        ] = block
    elif rand_loc == 2:
        output[
            image_size - margin - pat_size_x : image_size - margin,
            margin : margin + pat_size_y,
            :,
        ] = block
    elif rand_loc == 3:
        output[
            image_size - margin - pat_size_x : image_size - margin,
            image_size - margin - pat_size_y : image_size - margin,
            :,
        ] = block  # right bottom

    output[output > 1] = 1
    return output


class FrequencyDetector(nn.Module):
    def __init__(self, height, width):
        super(FrequencyDetector, self).__init__()

        self.num_classes = 2  # poisoned or clean
        self.height = height
        self.width = width
        self.softmax = nn.Softmax(dim=1)

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(128 * (self.height // 8) * (self.width // 8), self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x
