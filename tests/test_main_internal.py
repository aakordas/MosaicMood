import argparse
import runpy
import sys
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import main


class DummyParser:
    def error(
        self,
        message: str
    ) -> None:
        raise ValueError(
            message
        )


def test_run_video_mode_rejects_stdin() -> None:
    args = argparse.Namespace(
        image = r'-',
        mode = r'full',
        output = None,
        strip_size = 1,
        cluster_size = 10,
        cluster_reduced = False,
        video_reverse = False
    )

    with pytest.raises(
        ValueError,
        match = r'Video mode does not support stdin input.'
    ):
        main._run_video_mode(
            args,
            DummyParser()
        )


def test_run_video_mode_rejects_invalid_strip_size() -> None:
    args = argparse.Namespace(
        image = r'input.avi',
        mode = r'full',
        output = None,
        strip_size = 0,
        cluster_size = 10,
        cluster_reduced = False,
        video_reverse = False
    )

    with pytest.raises(
        ValueError,
        match = r'--strip-size must be >= 1'
    ):
        main._run_video_mode(
            args,
            DummyParser()
        )


def test_run_video_mode_rejects_invalid_cluster_size() -> None:
    args = argparse.Namespace(
        image = r'input.avi',
        mode = r'full',
        output = None,
        strip_size = 1,
        cluster_size = 0,
        cluster_reduced = False,
        video_reverse = False
    )

    with pytest.raises(
        ValueError,
        match = r'--cluster-size must be >= 1'
    ):
        main._run_video_mode(
            args,
            DummyParser()
        )


def test_run_video_mode_preview_when_output_missing(
    monkeypatch: Any,
    capsys
) -> None:
    calls = {
        r'preview': 0
    }

    def fake_video_request(
        *_args,
        **_kwargs
    ) -> SimpleNamespace:
        img = np.zeros(
            (1, 4, 3),
            dtype = np.uint8
        )
        return SimpleNamespace(
            timeline_img = img,
            frame_count = 4
        )

    def fake_preview(
        _img,
        title = None
    ):
        calls[r'preview'] += 1
        assert title == r'Video timeline preview'

    monkeypatch.setattr(
        main,
        r'process_video_request',
        fake_video_request
    )
    monkeypatch.setattr(
        main,
        r'preview_image_in_terminal',
        fake_preview
    )

    args = argparse.Namespace(
        image = r'input.avi',
        mode = r'full',
        output = None,
        strip_size = 1,
        cluster_size = 10,
        cluster_reduced = False,
        video_reverse = False
    )

    main._run_video_mode(
        args,
        DummyParser()
    )

    out = capsys.readouterr().out
    assert calls[r'preview'] == 1
    assert r'Frames processed: 4' in out


def test_run_video_mode_saves_output(
    monkeypatch: Any,
    tmp_path: Path,
    capsys
) -> None:
    saved = {
        r'path': None,
        r'shape': None
    }

    def fake_video_request(
        *_args,
        **_kwargs
    ) -> SimpleNamespace:
        img = np.zeros(
            (1, 5, 3),
            dtype = np.uint8
        )
        return SimpleNamespace(
            timeline_img = img,
            frame_count = 5
        )

    def fake_save(
        path,
        img_rgb
    ):
        saved[r'path'] = path
        saved[r'shape'] = img_rgb.shape

    monkeypatch.setattr(
        main,
        r'process_video_request',
        fake_video_request
    )
    monkeypatch.setattr(
        main,
        r'save_rgb_image',
        fake_save
    )

    output = tmp_path / r'out.png'

    args = argparse.Namespace(
        image = r'input.avi',
        mode = r'cluster',
        output = str(output),
        strip_size = 1,
        cluster_size = 3,
        cluster_reduced = True,
        video_reverse = True
    )

    main._run_video_mode(
        args,
        DummyParser()
    )

    out = capsys.readouterr().out
    assert saved[r'path'] == str(output)
    assert saved[r'shape'] == (1, 5, 3)
    assert r'Mode used per frame: cluster' in out


def test_main_runs_full_image_flow(
    monkeypatch: Any,
    sample_image_path: Path
) -> None:
    called = {
        r'print_average': 0
    }

    args = argparse.Namespace(
        image = str(sample_image_path),
        video = False,
        video_reverse = False,
        mode = r'full',
        output = None,
        strip_size = 1,
        cluster_size = 10,
        cluster_reduced = False
    )

    monkeypatch.setattr(
        main,
        r'parse_args',
        lambda *_args, **_kwargs: (argparse.ArgumentParser(), args)
    )

    def fake_print_average(
        _rgb
    ):
        called[r'print_average'] += 1

    monkeypatch.setattr(
        main,
        r'print_average_colour',
        fake_print_average
    )

    main.main()

    assert called[r'print_average'] == 1


def test_main_routes_video_flag_to_video_runner(
    monkeypatch: Any,
    sample_image_path: Path
) -> None:
    calls = {
        r'video_runner': 0
    }

    args = argparse.Namespace(
        image = str(sample_image_path),
        video = True,
        video_reverse = False,
        mode = r'full',
        output = None,
        strip_size = 1,
        cluster_size = 10,
        cluster_reduced = False
    )

    monkeypatch.setattr(
        main,
        r'parse_args',
        lambda *_args, **_kwargs: (argparse.ArgumentParser(), args)
    )

    def fake_video_runner(
        _args,
        _parser
    ):
        calls[r'video_runner'] += 1

    monkeypatch.setattr(
        main,
        r'_run_video_mode',
        fake_video_runner
    )

    main.main()

    assert calls[r'video_runner'] == 1


def test_main_row_preview_without_output(
    monkeypatch: Any,
    sample_image_path: Path,
    capsys
) -> None:
    calls = {
        r'preview': 0
    }

    args = argparse.Namespace(
        image = str(sample_image_path),
        video = False,
        video_reverse = False,
        mode = r'row',
        output = None,
        strip_size = 1,
        cluster_size = 10,
        cluster_reduced = False
    )

    monkeypatch.setattr(
        main,
        r'parse_args',
        lambda *_args, **_kwargs: (argparse.ArgumentParser(), args)
    )

    def fake_preview(
        _img,
        title = None
    ):
        calls[r'preview'] += 1
        assert title == r'Row mode preview'

    monkeypatch.setattr(
        main,
        r'preview_image_in_terminal',
        fake_preview
    )

    main.main()

    out = capsys.readouterr().out
    assert calls[r'preview'] == 1
    assert r'Preview only. Use --output to save the image.' in out


def test_main_column_mode_saves_output(
    monkeypatch: Any,
    sample_image_path: Path,
    tmp_path: Path,
    capsys
) -> None:
    saved = {
        r'path': None,
        r'shape': None
    }

    output = tmp_path / r'column.png'

    args = argparse.Namespace(
        image = str(sample_image_path),
        video = False,
        video_reverse = False,
        mode = r'column',
        output = str(output),
        strip_size = 1,
        cluster_size = 10,
        cluster_reduced = False
    )

    monkeypatch.setattr(
        main,
        r'parse_args',
        lambda *_args, **_kwargs: (argparse.ArgumentParser(), args)
    )

    def fake_save(
        path,
        img_rgb
    ):
        saved[r'path'] = path
        saved[r'shape'] = img_rgb.shape

    monkeypatch.setattr(
        main,
        r'save_rgb_image',
        fake_save
    )

    main.main()

    out = capsys.readouterr().out
    assert saved[r'path'] == str(output)
    assert saved[r'shape'][0] == 1
    assert r'Saved column average strip to' in out


def test_main_cluster_default_path_upscales(
    monkeypatch: Any,
    sample_image_path: Path,
    tmp_path: Path
) -> None:
    calls = {
        r'cluster_reduced': None
    }

    output = tmp_path / r'cluster.png'

    args = argparse.Namespace(
        image = str(sample_image_path),
        video = False,
        video_reverse = False,
        mode = r'cluster',
        output = str(output),
        strip_size = 1,
        cluster_size = 3,
        cluster_reduced = False
    )

    monkeypatch.setattr(
        main,
        r'parse_args',
        lambda *_args, **_kwargs: (argparse.ArgumentParser(), args)
    )

    def fake_process_image_request(
        **kwargs: Any
    ) -> SimpleNamespace:
        calls[r'cluster_reduced'] = kwargs[r'cluster_reduced']
        source = np.zeros(
            (6, 8, 3),
            dtype = np.uint8
        )
        result_img = np.zeros(
            (6, 8, 3),
            dtype = np.uint8
        )

        return SimpleNamespace(
            mode = r'cluster',
            source_img = source,
            average_rgb = None,
            result_img = result_img
        )

    monkeypatch.setattr(
        main,
        r'process_image_request',
        fake_process_image_request
    )

    main.main()

    assert calls[r'cluster_reduced'] is False


def test_main_cluster_reduced_skips_upscale(
    monkeypatch: Any,
    sample_image_path: Path,
    tmp_path: Path
) -> None:
    calls = {
        r'cluster_reduced': None
    }

    output = tmp_path / r'cluster_reduced.png'

    args = argparse.Namespace(
        image = str(sample_image_path),
        video = False,
        video_reverse = False,
        mode = r'cluster',
        output = str(output),
        strip_size = 1,
        cluster_size = 3,
        cluster_reduced = True
    )

    monkeypatch.setattr(
        main,
        r'parse_args',
        lambda *_args, **_kwargs: (argparse.ArgumentParser(), args)
    )

    def fake_process_image_request(
        **kwargs: Any
    ) -> SimpleNamespace:
        calls[r'cluster_reduced'] = kwargs[r'cluster_reduced']
        source = np.zeros(
            (6, 8, 3),
            dtype = np.uint8
        )
        result_img = np.zeros(
            (2, 3, 3),
            dtype = np.uint8
        )

        return SimpleNamespace(
            mode = r'cluster',
            source_img = source,
            average_rgb = None,
            result_img = result_img
        )

    monkeypatch.setattr(
        main,
        r'process_image_request',
        fake_process_image_request
    )

    main.main()

    assert calls[r'cluster_reduced'] is True


def test_main_rejects_invalid_strip_size(
    monkeypatch: Any,
    sample_image_path: Path
) -> None:
    args = argparse.Namespace(
        image = str(sample_image_path),
        video = False,
        video_reverse = False,
        mode = r'row',
        output = None,
        strip_size = 0,
        cluster_size = 10,
        cluster_reduced = False
    )

    monkeypatch.setattr(
        main,
        r'parse_args',
        lambda *_args, **_kwargs: (argparse.ArgumentParser(), args)
    )

    with pytest.raises(
        SystemExit
    ):
        main.main()


def test_main_rejects_invalid_cluster_size(
    monkeypatch: Any,
    sample_image_path: Path
) -> None:
    args = argparse.Namespace(
        image = str(sample_image_path),
        video = False,
        video_reverse = False,
        mode = r'cluster',
        output = None,
        strip_size = 1,
        cluster_size = 0,
        cluster_reduced = False
    )

    monkeypatch.setattr(
        main,
        r'parse_args',
        lambda *_args, **_kwargs: (argparse.ArgumentParser(), args)
    )

    with pytest.raises(
        SystemExit
    ):
        main.main()


def test_main_module_entrypoint_executes(
    monkeypatch: Any,
    repo_root: Path,
    tmp_path: Path
) -> None:
    image_path = tmp_path / r'entrypoint.png'
    img = np.zeros(
        (2, 2, 3),
        dtype = np.uint8
    )
    cv2.imwrite(
        str(image_path),
        img
    )

    monkeypatch.setattr(
        sys,
        r'argv',
        [
            r'main.py',
            str(image_path),
            r'--mode',
            r'full'
        ]
    )

    runpy.run_path(
        str(repo_root / r'main.py'),
        run_name = r'__main__'
    )
