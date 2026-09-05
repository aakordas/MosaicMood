import argparse
from collections.abc import Sequence
from cli_args import parse_args
from image_ops import save_rgb_image
from processing_service import process_image_request, process_video_request
from terminal_output import (
    print_average_colour,
    preview_image_in_terminal
)

def main(
    argv: Sequence[str] | None = None
) -> None:
    parser, args = parse_args(
        argv = argv
    )

    try:
        if args.video:
            _run_video_mode(
                args,
                parser
            )
            return

        result = process_image_request(
            input_path = args.image,
            mode = args.mode,
            strip_size = args.strip_size,
            cluster_size = args.cluster_size,
            cluster_reduced = args.cluster_reduced,
            show_progress = True
        )

        if result.mode == r'full':
            assert result.average_rgb is not None
            print_average_colour(
                result.average_rgb
            )

            return

        assert result.result_img is not None
        result_img = result.result_img
        source_img = result.source_img

        if args.output is None:
            preview_image_in_terminal(
                result_img,
                title = f"{result.mode.title()} mode preview"
            )

            print(
                f"Source size: {source_img.shape[1]}x{source_img.shape[0]} (W x H)"
            )
            print(
                f"Result size: {result_img.shape[1]}x{result_img.shape[0]} (W x H)"
            )
            print(
                r'Preview only. Use --output to save the image.'
            )

            return

        save_rgb_image(
            args.output,
            result_img
        )

        if result.mode == r'cluster':
            if args.cluster_reduced:
                print(
                    f"Saved cluster average image to {args.output}"
                )
            else:
                print(
                    f"Saved upscaled cluster average image to {args.output}"
                )
        else:
            print(
                f"Saved {result.mode} average strip to {args.output}"
            )
    except (ValueError, OSError) as exc:
        parser.exit(
            status = 2,
            message = f"Error: {exc}\n"
        )


def _run_video_mode(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser
) -> None:
    result = process_video_request(
        input_path = args.image,
        mode = args.mode,
        strip_size = args.strip_size,
        cluster_size = args.cluster_size,
        cluster_reduced = args.cluster_reduced,
        reverse = args.video_reverse,
        show_progress = True,
        progress_desc = f"Video {args.mode} mode"
    )

    timeline_img = result.timeline_img
    frame_count = result.frame_count

    if args.output is None:
        preview_image_in_terminal(
            timeline_img,
            title = r'Video timeline preview'
        )

        print(
            f"Frames processed: {frame_count}"
        )
        print(
            f"Timeline size: {timeline_img.shape[1]}x{timeline_img.shape[0]} (W x H)"
        )
        print(
            r'Preview only. Use --output to save the timeline image.'
        )
        return

    save_rgb_image(
        args.output,
        timeline_img
    )

    print(
        f"Saved video timeline image to {args.output}"
    )
    print(
        f"Frames processed: {frame_count}"
    )
    print(
        f"Mode used per frame: {args.mode}"
    )
    print(
        f"Timeline size: {timeline_img.shape[1]}x{timeline_img.shape[0]} (W x H)"
    )


if __name__ == r'__main__':
    main()
