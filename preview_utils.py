import cv2
import numpy as np
import numpy.typing as npt


ImageArray = npt.NDArray[np.uint8]


def fit_image_nearest(
    img_rgb: ImageArray,
    max_width: int,
    max_height: int
) -> ImageArray:
    height, width = img_rgb.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    out_width = max(
        1,
        int(width * scale)
    )
    out_height = max(
        1,
        int(height * scale)
    )

    return cv2.resize(
        img_rgb,
        (out_width, out_height),
        interpolation = cv2.INTER_NEAREST
    )


def stretch_timeline_nearest(
    timeline_img: ImageArray,
    target_height: int
) -> ImageArray:
    return cv2.resize(
        timeline_img,
        (timeline_img.shape[1], target_height),
        interpolation = cv2.INTER_NEAREST
    )
