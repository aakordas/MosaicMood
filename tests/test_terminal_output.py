import numpy as np
import numpy.typing as npt

from terminal_output import (
    colour_block,
    print_average_colour,
    preview_image_in_terminal,
    _resize_nearest_for_terminal,
    _render_row
)

ImageArray = npt.NDArray[np.uint8]


def test_colour_block_contains_truecolor_escape() -> None:
    block = colour_block(
        (1, 2, 3),
        width = 2
    )

    assert r'\x1b[48;2;1;2;3m' in repr(
        block
    )
    assert block.endswith(
        "\x1b[0m"
    )


def test_resize_nearest_for_terminal_bounds_output() -> None:
    img = np.zeros(
        (100, 200, 3),
        dtype = np.uint8
    )

    resized = _resize_nearest_for_terminal(
        img,
        max_width = 20,
        max_height = 10
    )

    assert resized.shape[1] <= 20
    assert resized.shape[0] <= 10


def test_render_row_emits_ansi_segments() -> None:
    row = np.array(
        [
            [255, 0, 0],
            [0, 255, 0]
        ],
        dtype = np.uint8
    )

    rendered = _render_row(
        row,
        pixel_width = 1
    )

    rendered_repr = repr(
        rendered
    )

    assert r'\x1b[48;2;255;0;0m' in rendered_repr
    assert r'\x1b[48;2;0;255;0m' in rendered_repr


def test_preview_image_in_terminal_prints_summary(
    capsys
) -> None:
    img = np.zeros(
        (2, 4, 3),
        dtype = np.uint8
    )
    img[:, :] = [10, 20, 30]

    preview_image_in_terminal(
        img,
        title = r'Test preview',
        pixel_width = 1,
        margin_lines = 2
    )

    output = capsys.readouterr().out

    assert r'Test preview' in output
    assert r'Preview size:' in output


def test_print_average_colour_outputs_text_and_block(
    capsys
) -> None:
    print_average_colour(
        (7, 8, 9)
    )

    output = capsys.readouterr().out

    assert r'Average colour (R, G, B): (7, 8, 9)' in output
    assert r'\x1b[48;2;7;8;9m' in repr(
        output
    )
