import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = r'Calculate average colour of an image.'
    )

    parser.add_argument(
        r'image',
        help = r'Path to input image/video file or "-" for stdin image bytes'
    )

    parser.add_argument(
        r'--video',
        action = r'store_true',
        help = r'Treat input as video and generate a frame-average timeline image using the selected mode per frame'
    )

    parser.add_argument(
        r'--video-reverse',
        action = r'store_true',
        help = r'Reverse frame order in video timeline output (last frame on the left)'
    )

    parser.add_argument(
        r'--mode',
        choices = [r'full', r'row', r'column', r'cluster'],
        default = r'full',
        help = r'Output mode: full average in terminal, row strip image, column strip image, or clustered average image'
    )

    parser.add_argument(
        r'--output',
        help = r'Output image path for row/column/cluster modes; if omitted, preview is printed in terminal'
    )

    parser.add_argument(
        r'--strip-size',
        type = int,
        default = 1,
        help = r'Width for row strip or height for column strip (default: 1 for strict output)'
    )

    parser.add_argument(
        r'--cluster-size',
        type = int,
        default = 10,
        help = r'Block size used for clustered average mode'
    )

    parser.add_argument(
        r'--cluster-reduced',
        action = r'store_true',
        help = r'Keep cluster output at reduced size instead of upscaling to original dimensions'
    )

    return parser


def parse_args(
    argv: Sequence[str] | None = None
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(
        args = argv
    )

    return parser, args
