import cv2
import io
import numpy as np
import numpy.typing as npt
import pytest
import sys
from typing import Any

from image_ops import (
    read_image,
    average_colour,
    average_colour_per_row,
    average_colour_per_column,
    create_row_average_strip,
    create_column_average_strip,
    create_cluster_average_image,
    upscale_image_nearest,
    save_rgb_image
)


ImageArray = npt.NDArray[np.uint8]


def test_read_image_from_path(
    sample_image_path: Any
) -> None:
    img = read_image(
        str(sample_image_path)
    )

    assert img.shape == (6, 8, 3)


def test_average_colour_returns_expected_rgb(
    sample_rgb_image: ImageArray
) -> None:
    bgr = cv2.cvtColor(
        sample_rgb_image,
        cv2.COLOR_RGB2BGR
    )

    result = average_colour(
        bgr
    )

    assert result == (110, 90, 110)


def test_row_and_column_average_shapes(
    sample_rgb_image: ImageArray
) -> None:
    bgr = cv2.cvtColor(
        sample_rgb_image,
        cv2.COLOR_RGB2BGR
    )

    row_colours = average_colour_per_row(
        bgr
    )
    col_colours = average_colour_per_column(
        bgr
    )

    assert row_colours.shape == (6, 3)
    assert col_colours.shape == (8, 3)


def test_strip_images_have_requested_thickness(
    sample_rgb_image: ImageArray
) -> None:
    bgr = cv2.cvtColor(
        sample_rgb_image,
        cv2.COLOR_RGB2BGR
    )

    row_strip = create_row_average_strip(
        bgr,
        strip_width = 3
    )
    col_strip = create_column_average_strip(
        bgr,
        strip_height = 2
    )

    assert row_strip.shape == (6, 3, 3)
    assert col_strip.shape == (2, 8, 3)


def test_cluster_reduced_and_upscaled_shapes(
    sample_rgb_image: ImageArray
) -> None:
    bgr = cv2.cvtColor(
        sample_rgb_image,
        cv2.COLOR_RGB2BGR
    )

    reduced = create_cluster_average_image(
        bgr,
        cluster_size = 3,
        show_progress = False
    )

    upscaled = upscale_image_nearest(
        reduced,
        target_width = bgr.shape[1],
        target_height = bgr.shape[0]
    )

    assert reduced.shape == (2, 3, 3)
    assert upscaled.shape == (6, 8, 3)


def test_save_rgb_image_roundtrip(
    tmp_path: Any,
    sample_rgb_image: ImageArray
) -> None:
    path = tmp_path / r'roundtrip.png'

    save_rgb_image(
        str(path),
        sample_rgb_image
    )

    loaded = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )
    loaded_rgb = cv2.cvtColor(
        loaded,
        cv2.COLOR_BGR2RGB
    )

    assert np.array_equal(
        loaded_rgb,
        sample_rgb_image
    )


def test_read_image_from_stdin_bytes(
    monkeypatch: Any,
    sample_rgb_image: ImageArray
) -> None:
    bgr = cv2.cvtColor(
        sample_rgb_image,
        cv2.COLOR_RGB2BGR
    )

    ok, encoded = cv2.imencode(
        r'.png',
        bgr
    )
    assert ok

    class FakeStdin:
        def __init__(
            self,
            data: bytes
        ) -> None:
            self.buffer = io.BytesIO(
                data
            )

    monkeypatch.setattr(
        sys,
        r'stdin',
        FakeStdin(
            encoded.tobytes()
        )
    )

    img = read_image(
        r'-'
    )

    assert img.shape == (6, 8, 3)


def test_read_image_raises_on_invalid_path() -> None:
    with pytest.raises(
        ValueError,
        match = r'Could not load image.'
    ):
        read_image(
            r'/definitely/not/a/file.png'
        )


def test_read_image_raises_on_empty_stdin(
    monkeypatch: Any
) -> None:
    class FakeStdin:
        def __init__(
            self
        ) -> None:
            self.buffer = io.BytesIO(
                b''
            )

    monkeypatch.setattr(
        sys,
        r'stdin',
        FakeStdin()
    )

    with pytest.raises(
        ValueError,
        match = r'No bytes were received from stdin'
    ):
        read_image(
            r'-'
        )


def test_read_image_raises_on_directory_path(
    tmp_path: Any
) -> None:
    with pytest.raises(
        ValueError,
        match = r'Input path is not a file:'
    ):
        read_image(
            str(tmp_path)
        )


def test_create_cluster_with_progress_enabled(
    monkeypatch: Any,
    sample_rgb_image: ImageArray
) -> None:
    import image_ops

    calls = {
        r'tqdm': 0
    }

    def fake_tqdm(
        iterable: Any,
        **_kwargs
    ) -> Any:
        calls[r'tqdm'] += 1
        return iterable

    monkeypatch.setattr(
        image_ops,
        r'tqdm',
        fake_tqdm
    )

    bgr = cv2.cvtColor(
        sample_rgb_image,
        cv2.COLOR_RGB2BGR
    )

    out = create_cluster_average_image(
        bgr,
        cluster_size = 3,
        show_progress = True,
        progress_desc = r'test'
    )

    assert out.shape == (2, 3, 3)
    assert calls[r'tqdm'] == 1


def test_save_rgb_image_raises_when_write_fails(
    monkeypatch: Any,
    tmp_path: Any,
    sample_rgb_image: ImageArray
) -> None:
    monkeypatch.setattr(
        cv2,
        r'imwrite',
        lambda *_args, **_kwargs: False
    )

    with pytest.raises(
        ValueError,
        match = r'Could not write image to:'
    ):
        save_rgb_image(
            str(tmp_path / r'will_fail.png'),
            sample_rgb_image
        )


def test_save_rgb_image_raises_when_output_directory_missing(
    tmp_path: Any,
    sample_rgb_image: ImageArray
) -> None:
    output = tmp_path / r'not_here' / r'image.png'

    with pytest.raises(
        ValueError,
        match = r'Output directory does not exist:'
    ):
        save_rgb_image(
            str(output),
            sample_rgb_image
        )


def test_save_rgb_image_raises_when_imwrite_throws(
    monkeypatch: Any,
    tmp_path: Any,
    sample_rgb_image: ImageArray
) -> None:
    def raising_imwrite(
        *_args: Any,
        **_kwargs: Any
    ) -> Any:
        raise RuntimeError(
            r'imwrite failed'
        )

    monkeypatch.setattr(
        cv2,
        r'imwrite',
        raising_imwrite
    )

    with pytest.raises(
        ValueError,
        match = r'OpenCV failed while encoding or writing the file'
    ):
        save_rgb_image(
            str(tmp_path / r'throws.png'),
            sample_rgb_image
        )


def test_row_and_column_averages_match_expected_values() -> None:
    # BGR input where channels are easy to reason about after BGR->RGB conversion.
    bgr = np.array(
        [
            [[0, 0, 10], [0, 0, 30]],
            [[0, 0, 50], [0, 0, 70]],
        ],
        dtype = np.uint8
    )

    row = average_colour_per_row(
        bgr
    )
    col = average_colour_per_column(
        bgr
    )

    assert row.tolist() == [[20, 0, 0], [60, 0, 0]]
    assert col.tolist() == [[30, 0, 0], [50, 0, 0]]


def test_cluster_average_handles_non_divisible_dimensions() -> None:
    # 3x3 image with cluster_size=2 produces a 2x2 reduced output.
    # Build BGR from target RGB values for clarity.
    rgb = np.array(
        [
            [[10, 0, 0], [30, 0, 0], [90, 0, 0]],
            [[50, 0, 0], [70, 0, 0], [110, 0, 0]],
            [[130, 0, 0], [150, 0, 0], [170, 0, 0]],
        ],
        dtype = np.uint8
    )
    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    reduced = create_cluster_average_image(
        bgr,
        cluster_size = 2,
        show_progress = False
    )

    # block(0,0): mean of [10,30,50,70] = 40
    # block(0,1): mean of [90,110] = 100
    # block(1,0): mean of [130,150] = 140
    # block(1,1): [170] = 170
    expected = np.array(
        [
            [[40, 0, 0], [100, 0, 0]],
            [[140, 0, 0], [170, 0, 0]],
        ],
        dtype = np.uint8
    )

    assert np.array_equal(
        reduced,
        expected
    )
