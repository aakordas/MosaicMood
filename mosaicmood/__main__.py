from collections.abc import Sequence

import main as legacy_main


def main(
    argv: Sequence[str] | None = None
) -> None:
    legacy_main.main(
        argv = argv
    )


if __name__ == r'__main__':
    main()
