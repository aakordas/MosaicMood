import cv2
import os
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import cast, Literal

from image_ops import (
    average_colour,
    create_row_average_strip,
    create_column_average_strip,
    create_cluster_average_image,
    upscale_image_nearest
)


RGBTuple = tuple[int, int, int]
ImageArray = npt.NDArray[np.uint8]
Mode = Literal['full', 'row', 'column', 'cluster']


def create_video_average_timeline_image(
    input_path: str,
    mode: Mode = r'full',
    strip_size: int = 1,
    cluster_size: int = 10,
    cluster_reduced: bool = False,
    reverse: bool = False,
    show_progress: bool = True,
    progress_desc: str = r'Processing video frames'
) -> tuple[ImageArray, int]:
    capture = cv2.VideoCapture(
        input_path
    )

    if not capture.isOpened():
        if not os.path.exists(input_path):
            raise ValueError(
                f"Could not open video: {input_path}. File not found. Check the path and permissions."
            )

        raise ValueError(
            f"Could not open video: {input_path}. The file may be corrupted, unsupported by OpenCV, or locked by another process."
        )

    colours = []

    frame_total_raw = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )
    frame_total: int | None = frame_total_raw if frame_total_raw > 0 else None

    progress = None
    if show_progress:
        progress = tqdm(
            total = frame_total,
            desc = progress_desc,
            unit = r'frame',
            leave = False
        )

    try:
        while True:
            ok, frame_raw = capture.read()  # type: ignore[var-annotated]

            if not ok:
                break

            frame: ImageArray = cast(
                ImageArray,
                frame_raw
            )

            processed_rgb = process_video_frame_with_mode(
                frame,
                mode = mode,
                strip_size = strip_size,
                cluster_size = cluster_size,
                cluster_reduced = cluster_reduced
            )

            colours.append(
                average_colour_rgb_image(
                    processed_rgb
                )
            )

            if progress is not None:
                progress.update(
                    1
                )
    finally:
        capture.release()

        if progress is not None:
            progress.close()

    frame_count = len(
        colours
    )

    if frame_count == 0:
        raise ValueError(
            r'Video has no readable frames. The file may be empty, corrupted, or encoded with an unsupported codec.'
        )

    if reverse:
        colours.reverse()

    timeline = np.array(
        colours,
        dtype = np.uint8
    )

    # 1-pixel tall timeline image: columns map to frame index.
    timeline_img = timeline[np.newaxis, :, :]

    return timeline_img, frame_count


def process_video_frame_with_mode(
    frame: ImageArray,
    mode: Mode,
    strip_size: int = 1,
    cluster_size: int = 10,
    cluster_reduced: bool = False
) -> ImageArray:
    if mode == r'full':
        avg = average_colour(
            frame
        )

        return np.array(
            [[avg]],
            dtype = np.uint8
        )

    if mode == r'row':
        return create_row_average_strip(
            frame,
            strip_width = strip_size
        )

    if mode == r'column':
        return create_column_average_strip(
            frame,
            strip_height = strip_size
        )

    if mode == r'cluster':
        cluster = create_cluster_average_image(
            frame,
            cluster_size = cluster_size
        )

        if cluster_reduced:
            return cluster

        height, width = frame.shape[:2]

        return upscale_image_nearest(
            cluster,
            target_width = width,
            target_height = height
        )

    raise ValueError(
        f"Unsupported mode: {mode}. Expected one of: full, row, column, cluster."
    )


def average_colour_rgb_image(
    img_rgb: ImageArray
) -> RGBTuple:
    mean = img_rgb.mean(
        axis = (0, 1)
    )

    rgb_values = tuple(
        int(x) for x in mean
    )
    assert len(rgb_values) == 3
    return rgb_values
