import pytest

from cli_args import (
    build_parser,
    parse_args
)


def test_build_parser_accepts_full_mode_minimum_args() -> None:
    parser = build_parser()

    args = parser.parse_args(
        args = [r'input.png']
    )

    assert args.image == r'input.png'
    assert args.mode == r'full'
    assert args.video is False


def test_parse_args_parses_video_reverse_and_cluster_flags() -> None:
    _parser, args = parse_args(
        argv = [
            r'input.avi',
            r'--video',
            r'--mode',
            r'cluster',
            r'--cluster-size',
            r'8',
            r'--cluster-reduced',
            r'--video-reverse'
        ]
    )

    assert args.image == r'input.avi'
    assert args.video is True
    assert args.mode == r'cluster'
    assert args.cluster_size == 8
    assert args.cluster_reduced is True
    assert args.video_reverse is True


def test_parse_args_rejects_invalid_mode() -> None:
    with pytest.raises(
        SystemExit
    ):
        parse_args(
            argv = [
                r'input.png',
                r'--mode',
                r'unknown'
            ]
        )
