"""Image transforms used for training augmentation."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from scipy.ndimage import rotate as scipy_rotate


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if cv2 is not None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            image,
            rot_mat,
            image.shape[1::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_DEFAULT,
        )
        return result

    result = scipy_rotate(image, angle, reshape=False, order=1, mode="constant", cval=0.0)
    if result.dtype != image.dtype:
        if np.issubdtype(image.dtype, np.integer):
            result = np.clip(result, 0, 255).astype(image.dtype)
        else:
            result = result.astype(image.dtype)
    return result


def rotate_aug(img: np.ndarray, angles=(0, 45, 90, 135, 180, 225, 270, 315)) -> list[np.ndarray]:
    return [rotate_image(img, angle) for angle in angles]


def hflip_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(image, axis=1))


def vflip_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(image, axis=0))


def random_rotate_aug(
    img: np.ndarray,
    rng: np.random.RandomState,
    n: int = 2,
    angle_min: float = 0.0,
    angle_max: float = 360.0,
) -> list[np.ndarray]:
    if n <= 0:
        return []
    angles = rng.uniform(float(angle_min), float(angle_max), size=int(n))
    return [rotate_image(img, float(a)) for a in angles]


def color_jitter_image(
    image: np.ndarray,
    rng: np.random.RandomState,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> np.ndarray:
    x = image.astype(np.float32)

    b = 1.0 + float(rng.uniform(-brightness, brightness))
    x = x * b

    c = 1.0 + float(rng.uniform(-contrast, contrast))
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * c + mean

    s = 1.0 + float(rng.uniform(-saturation, saturation))
    gray = (
        0.299 * x[..., 0]
        + 0.587 * x[..., 1]
        + 0.114 * x[..., 2]
    )[..., None]
    x = gray * (1.0 - s) + x * s

    x = np.clip(x, 0.0, 255.0)
    if np.issubdtype(image.dtype, np.integer):
        return x.astype(image.dtype)
    return (x / 255.0).astype(image.dtype)


def color_jitter_aug(img: np.ndarray, rng: np.random.RandomState, n: int = 1) -> list[np.ndarray]:
    if n <= 0:
        return []
    return [color_jitter_image(img, rng) for _ in range(int(n))]
