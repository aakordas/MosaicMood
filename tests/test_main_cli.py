import subprocess
import sys
from pathlib import Path

import cv2


def _run_cli(
    repo_root: Path,
    *args
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        r'main.py',
        *args
    ]

    return subprocess.run(
        command,
        cwd = str(repo_root),
        capture_output = True,
        text = True,
        check = False
    )


def _run_module_cli(
    repo_root: Path,
    *args
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        r'-m',
        r'mosaicmood',
        *args
    ]

    return subprocess.run(
        command,
        cwd = str(repo_root),
        capture_output = True,
        text = True,
        check = False
    )


def test_cli_full_mode_prints_average(
    repo_root: Path,
    sample_image_path: Path
) -> None:
    result = _run_cli(
        repo_root,
        str(sample_image_path),
        r'--mode',
        r'full'
    )

    assert result.returncode == 0
    assert r'Average colour (R, G, B):' in result.stdout


def test_cli_row_mode_saves_output(
    repo_root: Path,
    sample_image_path: Path,
    tmp_path: Path
) -> None:
    output = tmp_path / r'row.png'

    result = _run_cli(
        repo_root,
        str(sample_image_path),
        r'--mode',
        r'row',
        r'--output',
        str(output)
    )

    assert result.returncode == 0
    assert output.exists()

    img = cv2.imread(
        str(output),
        cv2.IMREAD_COLOR
    )
    assert img.shape[:2] == (6, 1)


def test_cli_cluster_preview_without_output(
    repo_root: Path,
    sample_image_path: Path
) -> None:
    result = _run_cli(
        repo_root,
        str(sample_image_path),
        r'--mode',
        r'cluster',
        r'--cluster-size',
        r'3'
    )

    assert result.returncode == 0
    assert r'Preview only. Use --output to save the image.' in result.stdout


def test_cli_video_timeline_output(
    repo_root: Path,
    sample_video_path: Path,
    tmp_path: Path
) -> None:
    output = tmp_path / r'timeline.png'

    result = _run_cli(
        repo_root,
        str(sample_video_path),
        r'--video',
        r'--mode',
        r'full',
        r'--output',
        str(output)
    )

    assert result.returncode == 0
    assert output.exists()

    img = cv2.imread(
        str(output),
        cv2.IMREAD_COLOR
    )
    assert img.shape[:2] == (1, 4)


def test_cli_video_stdin_is_rejected(
    repo_root: Path
) -> None:
    result = _run_cli(
        repo_root,
        r'-',
        r'--video',
        r'--mode',
        r'full'
    )

    assert result.returncode != 0
    assert r'Video mode does not support stdin input.' in result.stderr


def test_cli_image_load_error_is_user_friendly(
    repo_root: Path
) -> None:
    result = _run_cli(
        repo_root,
        r'/definitely/not/a/real/file.png',
        r'--mode',
        r'full'
    )

    assert result.returncode == 2
    assert r'Error: Could not load image. Input file was not found:' in result.stderr


def test_cli_save_error_is_user_friendly(
    repo_root: Path,
    sample_image_path: Path,
    tmp_path: Path
) -> None:
    output = tmp_path / r'missing_dir' / r'out.png'

    result = _run_cli(
        repo_root,
        str(sample_image_path),
        r'--mode',
        r'row',
        r'--output',
        str(output)
    )

    assert result.returncode == 2
    assert r'Error: Could not write image to:' in result.stderr
    assert r'Output directory does not exist' in result.stderr


def test_cli_video_missing_path_error_is_user_friendly(
    repo_root: Path
) -> None:
    result = _run_cli(
        repo_root,
        r'/definitely/not/a/real/video.avi',
        r'--video',
        r'--mode',
        r'full'
    )

    assert result.returncode == 2
    assert r'Error: Could not open video:' in result.stderr
    assert r'File not found' in result.stderr


def test_cli_video_invalid_file_error_is_user_friendly(
    repo_root: Path,
    tmp_path: Path
) -> None:
    fake_video = tmp_path / r'not_a_video.txt'
    fake_video.write_text(
        r'this is not a video file'
    )

    result = _run_cli(
        repo_root,
        str(fake_video),
        r'--video',
        r'--mode',
        r'full'
    )

    assert result.returncode == 2
    assert r'Error: Could not open video:' in result.stderr
    assert r'unsupported by OpenCV' in result.stderr


def test_module_entrypoint_full_mode_prints_average(
    repo_root: Path,
    sample_image_path: Path
) -> None:
    result = _run_module_cli(
        repo_root,
        str(sample_image_path),
        r'--mode',
        r'full'
    )

    assert result.returncode == 0
    assert r'Average colour (R, G, B):' in result.stdout
