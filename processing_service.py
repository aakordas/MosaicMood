from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from image_ops import (
    average_colour,
    create_cluster_average_image,
    create_column_average_strip,
    create_row_average_strip,
    read_image,
    upscale_image_nearest,
)
from video_ops import create_video_average_timeline_image


RGBTuple = tuple[int, int, int]
ImageArray = npt.NDArray[np.uint8]
Mode = Literal['full', 'row', 'column', 'cluster']


@dataclass(frozen = True)
class ImageProcessResult:
    mode: Mode
    source_img: ImageArray
    average_rgb: RGBTuple | None
    result_img: ImageArray | None


@dataclass(frozen = True)
class VideoProcessResult:
    timeline_img: ImageArray
    frame_count: int


def normalize_mode(
    mode: str
) -> Mode:
    if mode == 'full':
        return 'full'
    if mode == 'row':
        return 'row'
    if mode == 'column':
        return 'column'
    if mode == 'cluster':
        return 'cluster'

    raise ValueError(
        f'Unsupported mode: {mode}. Expected one of: full, row, column, cluster.'
    )


def validate_common_sizes(
    strip_size: int,
    cluster_size: int
) -> None:
    if strip_size < 1:
        raise ValueError(
            '--strip-size must be >= 1'
        )

    if cluster_size < 1:
        raise ValueError(
            '--cluster-size must be >= 1'
        )


def validate_video_input(
    input_path: str
) -> None:
    if input_path == '-':
        raise ValueError(
            'Video mode does not support stdin input.'
        )


def process_image_request(
    input_path: str,
    mode: str,
    strip_size: int,
    cluster_size: int,
    cluster_reduced: bool,
    show_progress: bool = False
) -> ImageProcessResult:
    selected_mode = normalize_mode(mode)
    validate_common_sizes(
        strip_size,
        cluster_size
    )

    img = read_image(
        input_path
    )

    if selected_mode == 'full':
        if show_progress:
            with tqdm(
                total = 1,
                desc = 'Image full average',
                unit = 'step',
                leave = False
            ) as progress:
                avg = average_colour(
                    img
                )
                progress.update(
                    1
                )
        else:
            avg = average_colour(
                img
            )

        return ImageProcessResult(
            mode = selected_mode,
            source_img = img,
            average_rgb = avg,
            result_img = None
        )

    if selected_mode == 'row':
        if show_progress:
            with tqdm(
                total = 1,
                desc = 'Image row mode',
                unit = 'step',
                leave = False
            ) as progress:
                result_img = create_row_average_strip(
                    img,
                    strip_width = strip_size
                )
                progress.update(
                    1
                )
        else:
            result_img = create_row_average_strip(
                img,
                strip_width = strip_size
            )

        return ImageProcessResult(
            mode = selected_mode,
            source_img = img,
            average_rgb = None,
            result_img = result_img
        )

    if selected_mode == 'column':
        if show_progress:
            with tqdm(
                total = 1,
                desc = 'Image column mode',
                unit = 'step',
                leave = False
            ) as progress:
                result_img = create_column_average_strip(
                    img,
                    strip_height = strip_size
                )
                progress.update(
                    1
                )
        else:
            result_img = create_column_average_strip(
                img,
                strip_height = strip_size
            )

        return ImageProcessResult(
            mode = selected_mode,
            source_img = img,
            average_rgb = None,
            result_img = result_img
        )

    result_img = create_cluster_average_image(
        img,
        cluster_size = cluster_size,
        show_progress = show_progress,
        progress_desc = 'Image cluster mode'
    )

    if not cluster_reduced:
        height, width = img.shape[:2]
        if show_progress:
            with tqdm(
                total = 1,
                desc = 'Upscaling cluster image',
                unit = 'step',
                leave = False
            ) as progress:
                result_img = upscale_image_nearest(
                    result_img,
                    target_width = width,
                    target_height = height
                )
                progress.update(
                    1
                )
        else:
            result_img = upscale_image_nearest(
                result_img,
                target_width = width,
                target_height = height
            )

    return ImageProcessResult(
        mode = selected_mode,
        source_img = img,
        average_rgb = None,
        result_img = result_img
    )


def process_video_request(
    input_path: str,
    mode: str,
    strip_size: int,
    cluster_size: int,
    cluster_reduced: bool,
    reverse: bool,
    show_progress: bool,
    progress_desc: str
) -> VideoProcessResult:
    selected_mode = normalize_mode(
        mode
    )
    validate_common_sizes(
        strip_size,
        cluster_size
    )
    validate_video_input(
        input_path
    )

    timeline_img, frame_count = create_video_average_timeline_image(
        input_path,
        mode = selected_mode,
        strip_size = strip_size,
        cluster_size = cluster_size,
        cluster_reduced = cluster_reduced,
        reverse = reverse,
        show_progress = show_progress,
        progress_desc = progress_desc
    )

    return VideoProcessResult(
        timeline_img = timeline_img,
        frame_count = frame_count
    )
