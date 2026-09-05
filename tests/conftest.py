from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest


ImageArray = npt.NDArray[np.uint8]


def pytest_collection_modifyitems(
    items: list[Any]
) -> None:
    for item in items:
        node_path = str(
            item.fspath
        )

        if node_path.endswith(
            r'test_main_cli.py'
        ):
            item.add_marker(
                pytest.mark.integration
            )
            continue

        if node_path.endswith(
            r'test_main_internal.py'
        ):
            item.add_marker(
                pytest.mark.integration
            )
            continue

        item.add_marker(
            pytest.mark.unit
        )


@pytest.fixture
def sample_rgb_image() -> ImageArray:
    img = np.zeros(
        (6, 8, 3),
        dtype = np.uint8
    )

    img[:, :4] = (200, 40, 10)
    img[:, 4:] = (20, 140, 210)

    return img


@pytest.fixture
def sample_image_path(
    tmp_path: Path,
    sample_rgb_image: ImageArray
) -> Path:
    path = tmp_path / r'sample.png'

    bgr = cv2.cvtColor(
        sample_rgb_image,
        cv2.COLOR_RGB2BGR
    )

    saved = cv2.imwrite(
        str(path),
        bgr
    )
    if not saved:
        raise RuntimeError(
            r'Could not create test image file.'
        )

    return path


@pytest.fixture
def sample_video_path(
    tmp_path: Path
) -> Path:
    path = tmp_path / r'sample.avi'

    width = 10
    height = 6
    fps = 3.0

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*r'MJPG'),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            r'Could not create test video writer.'
        )

    frames_bgr = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255)
    ]

    for bgr in frames_bgr:
        frame = np.zeros(
            (height, width, 3),
            dtype = np.uint8
        )
        frame[:, :] = bgr
        writer.write(
            frame
        )

    writer.release()

    return path


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
