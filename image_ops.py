import sys
import os
import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import cast


RGBTuple = tuple[int, int, int]
ImageArray = npt.NDArray[np.uint8]
RowColumnAverageArray = npt.NDArray[np.uint8]


def read_image(
    path: str
) -> ImageArray:
    if path == r'-':
        image_bytes = sys.stdin.buffer.read()

        if len(image_bytes) == 0:
            raise ValueError(
                r'Could not load image. No bytes were received from stdin. '
                r'Pipe encoded image data (for example PNG/JPG bytes) into the command.'
            )

        image_array = np.frombuffer(
            image_bytes,
            dtype = np.uint8
        )

        try:
            img = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )
        except Exception as exc:
            raise ValueError(
                r'Could not load image. Failed to decode stdin bytes as an image. '
                r'Ensure the input stream is a valid encoded image format.'
            ) from exc

        if img is None:
            raise ValueError(
                r'Could not load image. Failed to decode stdin bytes as an image. '
                r'Ensure the input stream is a valid encoded image format.'
            )
    else:
        if not os.path.exists(path):
            raise ValueError(
                f"Could not load image. Input file was not found: {path}"
            )

        if not os.path.isfile(path):
            raise ValueError(
                f"Could not load image. Input path is not a file: {path}"
            )

        try:
            img = cv2.imread(
                path,
                cv2.IMREAD_COLOR
            )
        except Exception as exc:
            raise ValueError(
                f"Could not load image. OpenCV failed to read: {path}"
            ) from exc

    if img is None:
        raise ValueError(
            r'Could not load image. The file may be corrupted or an unsupported format.'
        )

    return cast(ImageArray, img)


def average_colour(
    img: ImageArray
) -> RGBTuple:
    img_rgb: ImageArray = cast(
        ImageArray,
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    )

    mean = img_rgb.mean(
        axis = (0, 1)
    )

    rgb_values = tuple(
        int(x) for x in mean
    )
    assert len(rgb_values) == 3
    return rgb_values


def average_colour_per_row(
    img: ImageArray
) -> RowColumnAverageArray:
    img_rgb: ImageArray = cast(
        ImageArray,
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    )

    return cast(
        RowColumnAverageArray,
        img_rgb.mean(
            axis = 1
        ).astype(
            np.uint8
        )
    )


def average_colour_per_column(
    img: ImageArray
) -> RowColumnAverageArray:
    img_rgb: ImageArray = cast(
        ImageArray,
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    )

    return cast(
        RowColumnAverageArray,
        img_rgb.mean(
            axis = 0
        ).astype(
            np.uint8
        )
    )


def create_row_average_strip(
    img: ImageArray,
    strip_width: int = 64
) -> ImageArray:
    row_colours = average_colour_per_row(
        img
    )

    return np.repeat(
        row_colours[:, np.newaxis, :],
        repeats = strip_width,
        axis = 1
    )


def create_column_average_strip(
    img: ImageArray,
    strip_height: int = 64
) -> ImageArray:
    column_colours = average_colour_per_column(
        img
    )

    return np.repeat(
        column_colours[np.newaxis, :, :],
        repeats = strip_height,
        axis = 0
    )


def create_cluster_average_image(
    img: ImageArray,
    cluster_size: int = 10,
    show_progress: bool = False,
    progress_desc: str = r'Processing cluster blocks'
) -> ImageArray:
    img_rgb: ImageArray = cast(
        ImageArray,
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    )

    height, width, _ = img_rgb.shape
    out_height = (height + cluster_size - 1) // cluster_size
    out_width = (width + cluster_size - 1) // cluster_size

    out_img = np.zeros(
        (out_height, out_width, 3),
        dtype = np.uint8
    )

    row_starts: range | tqdm[int] = range(
        0,
        height,
        cluster_size
    )

    if show_progress:
        row_starts = tqdm(
            row_starts,
            desc = progress_desc,
            unit = r'row',
            leave = False
        )

    for row_idx, row_start in enumerate(row_starts):
        row_end = min(
            row_start + cluster_size,
            height
        )

        for col_idx, col_start in enumerate(range(0, width, cluster_size)):
            col_end = min(
                col_start + cluster_size,
                width
            )

            block = img_rgb[
                row_start:row_end,
                col_start:col_end
            ]

            out_img[row_idx, col_idx] = block.mean(
                axis = (0, 1)
            ).astype(
                np.uint8
            )

    return out_img


def upscale_image_nearest(
    img_rgb: ImageArray,
    target_width: int,
    target_height: int
) -> ImageArray:
    return cv2.resize(
        img_rgb,
        (target_width, target_height),
        interpolation = cv2.INTER_NEAREST
    )


def save_rgb_image(
    path: str,
    img_rgb: ImageArray
) -> None:
    output_dir = os.path.dirname(path)
    if output_dir != r'' and not os.path.isdir(output_dir):
        raise ValueError(
            f"Could not write image to: {path}. Output directory does not exist: {output_dir}"
        )

    img_bgr: ImageArray = cast(
        ImageArray,
        cv2.cvtColor(
            img_rgb,
            cv2.COLOR_RGB2BGR
        )
    )

    try:
        saved = cv2.imwrite(
            path,
            img_bgr
        )
    except Exception as exc:
        raise ValueError(
            f"Could not write image to: {path}. OpenCV failed while encoding or writing the file."
        ) from exc

    if not saved:
        raise ValueError(
            f"Could not write image to: {path}. Check write permissions and use a supported extension like .png or .jpg."
        )
