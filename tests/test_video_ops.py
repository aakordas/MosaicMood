import numpy as np
import numpy.typing as npt
import pytest
from typing import Any

import video_ops
from video_ops import (
    create_video_average_timeline_image,
    process_video_frame_with_mode,
    average_colour_rgb_image
)


ImageArray = npt.NDArray[np.uint8]


def _channel_close(
    observed: list[int],
    expected: list[int],
    tolerance = 3
) -> bool:
    return all(
        abs(int(a) - int(b)) <= tolerance
        for a, b in zip(observed, expected)
    )


def test_create_video_timeline_full_mode(
    sample_video_path: Any
) -> None:
    timeline, frame_count = create_video_average_timeline_image(
        str(sample_video_path),
        mode = r'full',
        show_progress = False
    )

    assert frame_count == 4
    assert timeline.shape == (1, 4, 3)


def test_create_video_timeline_reverse_swaps_order(
    sample_video_path: Any
) -> None:
    forward, _ = create_video_average_timeline_image(
        str(sample_video_path),
        mode = r'full',
        reverse = False,
        show_progress = False
    )

    reverse, _ = create_video_average_timeline_image(
        str(sample_video_path),
        mode = r'full',
        reverse = True,
        show_progress = False
    )

    assert np.array_equal(
        forward[0, 0],
        reverse[0, -1]
    )
    assert np.array_equal(
        forward[0, -1],
        reverse[0, 0]
    )


def test_process_video_frame_reuses_modes() -> None:
    frame = np.zeros(
        (8, 12, 3),
        dtype = np.uint8
    )
    frame[:, :6] = (20, 80, 220)
    frame[:, 6:] = (200, 40, 30)

    full = process_video_frame_with_mode(
        frame,
        mode = r'full'
    )
    row = process_video_frame_with_mode(
        frame,
        mode = r'row',
        strip_size = 1
    )
    col = process_video_frame_with_mode(
        frame,
        mode = r'column',
        strip_size = 1
    )
    clu = process_video_frame_with_mode(
        frame,
        mode = r'cluster',
        cluster_size = 3,
        cluster_reduced = False
    )

    assert full.shape == (1, 1, 3)
    assert row.shape == (8, 1, 3)
    assert col.shape == (1, 12, 3)
    assert clu.shape == (8, 12, 3)


def test_average_colour_rgb_image() -> None:
    rgb = np.array(
        [
            [[10, 20, 30], [30, 40, 50]],
            [[50, 60, 70], [70, 80, 90]]
        ],
        dtype = np.uint8
    )

    avg = average_colour_rgb_image(
        rgb
    )

    assert avg == (40, 50, 60)


def test_video_timeline_first_and_last_colours_close_to_expected(
    sample_video_path: Any
) -> None:
    timeline, _ = create_video_average_timeline_image(
        str(sample_video_path),
        mode = r'full',
        show_progress = False
    )

    first = timeline[0, 0].tolist()
    last = timeline[0, -1].tolist()

    assert _channel_close(
        first,
        [255, 0, 0]
    )
    assert _channel_close(
        last,
        [255, 255, 0]
    )


def test_video_timeline_invalid_path_raises() -> None:
    with pytest.raises(
        ValueError,
        match = r'File not found'
    ):
        create_video_average_timeline_image(
            r'/nonexistent/file.avi',
            show_progress = False
        )


def test_video_timeline_existing_unopenable_file_raises(
    tmp_path: Any
) -> None:
    fake_video = tmp_path / r'not_a_video.txt'
    fake_video.write_text(
        r'this is not a valid video stream'
    )

    with pytest.raises(
        ValueError,
        match = r'The file may be corrupted, unsupported by OpenCV, or locked by another process'
    ):
        create_video_average_timeline_image(
            str(fake_video),
            show_progress = False
        )


def test_video_timeline_with_progress_unknown_total(
    monkeypatch: Any
) -> None:
    class FakeCapture:
        def __init__(
            self
        ) -> None:
            self.frames = [
                np.zeros((2, 2, 3), dtype = np.uint8),
                np.zeros((2, 2, 3), dtype = np.uint8)
            ]
            self.idx = 0

        def isOpened(
            self
        ) -> bool:
            return True

        def get(
            self,
            _prop: int
        ) -> int:
            return 0

        def read(
            self
        ) -> tuple[bool, ImageArray | None]:
            if self.idx >= len(self.frames):
                return False, None

            frame = self.frames[self.idx]
            self.idx += 1
            return True, frame

        def release(
            self
        ) -> None:
            return None

    class FakeProgress:
        def __init__(
            self
        ) -> None:
            self.updated = 0
            self.closed = False

        def update(
            self,
            n: int
        ) -> None:
            self.updated += n

        def close(
            self
        ) -> None:
            self.closed = True

    progress = FakeProgress()

    monkeypatch.setattr(
        video_ops.cv2,
        r'VideoCapture',
        lambda _path: FakeCapture()
    )

    def fake_tqdm(
        total = None,
        **_kwargs
    ) -> FakeProgress:
        assert total is None
        return progress

    monkeypatch.setattr(
        video_ops,
        r'tqdm',
        fake_tqdm
    )

    timeline, count = create_video_average_timeline_image(
        r'fake.avi',
        show_progress = True
    )

    assert timeline.shape == (1, 2, 3)
    assert count == 2
    assert progress.updated == 2
    assert progress.closed is True


def test_video_timeline_raises_on_empty_video(
    monkeypatch: Any
) -> None:
    class EmptyCapture:
        def isOpened(
            self
        ) -> bool:
            return True

        def get(
            self,
            _prop: int
        ) -> int:
            return 10

        def read(
            self
        ) -> tuple[bool, None]:
            return False, None

        def release(
            self
        ) -> None:
            return None

    monkeypatch.setattr(
        video_ops.cv2,
        r'VideoCapture',
        lambda _path: EmptyCapture()
    )

    with pytest.raises(
        ValueError,
        match = r'Video has no readable frames.'
    ):
        create_video_average_timeline_image(
            r'empty.avi',
            show_progress = False
        )


def test_process_video_frame_cluster_reduced_shape() -> None:
    frame = np.zeros(
        (8, 12, 3),
        dtype = np.uint8
    )
    frame[:, :6] = (20, 80, 220)
    frame[:, 6:] = (200, 40, 30)

    reduced = process_video_frame_with_mode(
        frame,
        mode = r'cluster',
        cluster_size = 3,
        cluster_reduced = True
    )

    assert reduced.shape == (3, 4, 3)
