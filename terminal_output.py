import shutil
import numpy as np
import numpy.typing as npt
from preview_utils import fit_image_nearest


RGBTuple = tuple[int, int, int]
ImageArray = npt.NDArray[np.uint8]
RowArray = npt.NDArray[np.uint8]


def colour_block(
    rgb: RGBTuple,
    width: int = 4
) -> str:
    r, g, b = rgb

    return f"\033[48;2;{r};{g};{b}m{' ' * width}\033[0m"


def print_average_colour(
    rgb: RGBTuple
) -> None:
    print(
        f"Average colour (R, G, B): {rgb}",
        end = r' '
    )

    print(
        colour_block(
            rgb
        )
    )


def preview_image_in_terminal(
    img_rgb: ImageArray,
    title: str | None = None,
    pixel_width: int = 2,
    margin_lines: int = 6
) -> None:
    term_size = shutil.get_terminal_size(
        fallback = (80, 24)
    )

    max_width = max(
        1,
        term_size.columns // pixel_width
    )
    max_height = max(
        1,
        term_size.lines - margin_lines
    )

    preview = _resize_nearest_for_terminal(
        img_rgb,
        max_width = max_width,
        max_height = max_height
    )

    height, width = preview.shape[:2]

    if title is not None:
        print(
            title
        )

    for row in preview:
        print(
            _render_row(
                row,
                pixel_width = pixel_width
            )
        )

    print(
        f"Preview size: {width}x{height} (W x H)"
    )


def _resize_nearest_for_terminal(
    img_rgb: ImageArray,
    max_width: int,
    max_height: int
) -> ImageArray:
    return fit_image_nearest(
        img_rgb,
        max_width = max_width,
        max_height = max_height
    )


def _render_row(
    row: RowArray,
    pixel_width: int = 2
) -> str:
    segments = []

    for r, g, b in row:
        segments.append(
            f"\033[48;2;{int(r)};{int(g)};{int(b)}m{' ' * pixel_width}\033[0m"
        )

    return r''.join(
        segments
    )
