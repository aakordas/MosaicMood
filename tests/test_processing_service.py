import numpy as np
import pytest
from typing import Any

from image_ops import (
    average_colour,
    create_cluster_average_image,
    create_column_average_strip,
    create_row_average_strip,
    upscale_image_nearest,
)
import processing_service
from processing_service import (
    normalize_mode,
    process_image_request,
    process_video_request,
    validate_common_sizes,
)


def test_normalize_mode_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match = r'Unsupported mode:'
    ):
        normalize_mode(
            'not-a-mode'
        )


def test_validate_common_sizes_rejects_invalid_values() -> None:
    with pytest.raises(
        ValueError,
        match = r'--strip-size must be >= 1'
    ):
        validate_common_sizes(
            strip_size = 0,
            cluster_size = 10
        )

    with pytest.raises(
        ValueError,
        match = r'--cluster-size must be >= 1'
    ):
        validate_common_sizes(
            strip_size = 1,
            cluster_size = 0
        )


def test_process_image_request_full_matches_average(
    sample_image_path
) -> None:
    result = process_image_request(
        input_path = str(sample_image_path),
        mode = 'full',
        strip_size = 1,
        cluster_size = 10,
        cluster_reduced = False,
        show_progress = False
    )

    assert result.mode == 'full'
    assert result.result_img is None
    assert result.average_rgb == average_colour(result.source_img)


def test_process_image_request_row_matches_direct_op(
    sample_image_path
) -> None:
    result = process_image_request(
        input_path = str(sample_image_path),
        mode = 'row',
        strip_size = 2,
        cluster_size = 10,
        cluster_reduced = False,
        show_progress = False
    )

    assert result.mode == 'row'
    assert result.average_rgb is None
    assert result.result_img is not None

    direct = create_row_average_strip(
        result.source_img,
        strip_width = 2
    )

    assert np.array_equal(
        result.result_img,
        direct
    )


def test_process_image_request_column_matches_direct_op(
    sample_image_path
) -> None:
    result = process_image_request(
        input_path = str(sample_image_path),
        mode = 'column',
        strip_size = 2,
        cluster_size = 10,
        cluster_reduced = False,
        show_progress = False
    )

    assert result.mode == 'column'
    assert result.average_rgb is None
    assert result.result_img is not None

    direct = create_column_average_strip(
        result.source_img,
        strip_height = 2
    )

    assert np.array_equal(
        result.result_img,
        direct
    )


def test_process_image_request_cluster_reduced_matches_direct_op(
    sample_image_path
) -> None:
    result = process_image_request(
        input_path = str(sample_image_path),
        mode = 'cluster',
        strip_size = 1,
        cluster_size = 3,
        cluster_reduced = True,
        show_progress = False
    )

    assert result.mode == 'cluster'
    assert result.average_rgb is None
    assert result.result_img is not None

    direct = create_cluster_average_image(
        result.source_img,
        cluster_size = 3,
        show_progress = False
    )

    assert np.array_equal(
        result.result_img,
        direct
    )


def test_process_image_request_cluster_upscaled_matches_direct_op(
    sample_image_path
) -> None:
    result = process_image_request(
        input_path = str(sample_image_path),
        mode = 'cluster',
        strip_size = 1,
        cluster_size = 3,
        cluster_reduced = False,
        show_progress = False
    )

    assert result.mode == 'cluster'
    assert result.average_rgb is None
    assert result.result_img is not None

    reduced = create_cluster_average_image(
        result.source_img,
        cluster_size = 3,
        show_progress = False
    )
    height, width = result.source_img.shape[:2]
    direct = upscale_image_nearest(
        reduced,
        target_width = width,
        target_height = height
    )

    assert np.array_equal(
        result.result_img,
        direct
    )


def test_process_video_request_rejects_stdin_input() -> None:
    with pytest.raises(
        ValueError,
        match = r'Video mode does not support stdin input.'
    ):
        process_video_request(
            input_path = '-',
            mode = 'full',
            strip_size = 1,
            cluster_size = 10,
            cluster_reduced = False,
            reverse = False,
            show_progress = False,
            progress_desc = 'Video test'
        )


def test_process_video_request_forwards_arguments(
    monkeypatch: Any
) -> None:
    captured: dict[str, Any] = {}
    timeline = np.zeros(
        (1, 2, 3),
        dtype = np.uint8
    )

    def fake_video_timeline(
        input_path: str,
        mode: str,
        strip_size: int,
        cluster_size: int,
        cluster_reduced: bool,
        reverse: bool,
        show_progress: bool,
        progress_desc: str
    ) -> tuple[np.ndarray[Any, Any], int]:
        captured.update(
            {
                'input_path': input_path,
                'mode': mode,
                'strip_size': strip_size,
                'cluster_size': cluster_size,
                'cluster_reduced': cluster_reduced,
                'reverse': reverse,
                'show_progress': show_progress,
                'progress_desc': progress_desc,
            }
        )
        return timeline, 2

    monkeypatch.setattr(
        processing_service,
        'create_video_average_timeline_image',
        fake_video_timeline
    )

    result = process_video_request(
        input_path = 'video.avi',
        mode = 'cluster',
        strip_size = 5,
        cluster_size = 7,
        cluster_reduced = True,
        reverse = True,
        show_progress = False,
        progress_desc = 'Custom progress'
    )

    assert result.frame_count == 2
    assert np.array_equal(
        result.timeline_img,
        timeline
    )
    assert captured == {
        'input_path': 'video.avi',
        'mode': 'cluster',
        'strip_size': 5,
        'cluster_size': 7,
        'cluster_reduced': True,
        'reverse': True,
        'show_progress': False,
        'progress_desc': 'Custom progress',
    }
