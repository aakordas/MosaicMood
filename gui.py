import base64
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from image_ops import save_rgb_image
from preview_utils import fit_image_nearest, stretch_timeline_nearest
from processing_service import (
    process_image_request,
    process_video_request,
)


RGBTuple = tuple[int, int, int]
ImageArray = npt.NDArray[np.uint8]


class MosaicMoodApp:
    def __init__(
        self,
        root: tk.Tk
    ) -> None:
        self.root = root
        self.root.title(
            r'MosaicMood'
        )
        self.root.geometry(
            r'980x640'
        )

        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.mode_var = tk.StringVar(
            value = r'full'
        )
        self.video_var = tk.BooleanVar(
            value = False
        )
        self.strip_size_var = tk.IntVar(
            value = 1
        )
        self.cluster_size_var = tk.IntVar(
            value = 10
        )
        self.cluster_reduced_var = tk.BooleanVar(
            value = False
        )
        self.video_reverse_var = tk.BooleanVar(
            value = False
        )
        self.result_text_var = tk.StringVar(
            value = r'Result: waiting'
        )

        self.preview_image: tk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(
        self
    ) -> None:
        frame = ttk.Frame(
            self.root,
            padding = 12
        )
        frame.pack(
            fill = tk.BOTH,
            expand = True
        )

        frame.columnconfigure(
            1,
            weight = 1
        )

        ttk.Label(
            frame,
            text = r'Input path'
        ).grid(
            row = 0,
            column = 0,
            sticky = tk.W,
            padx = 4,
            pady = 4
        )

        ttk.Entry(
            frame,
            textvariable = self.input_path_var
        ).grid(
            row = 0,
            column = 1,
            sticky = tk.EW,
            padx = 4,
            pady = 4
        )

        ttk.Button(
            frame,
            text = r'Browse Input',
            command = self._browse_input
        ).grid(
            row = 0,
            column = 2,
            sticky = tk.E,
            padx = 4,
            pady = 4
        )

        ttk.Label(
            frame,
            text = r'Output path (optional)'
        ).grid(
            row = 1,
            column = 0,
            sticky = tk.W,
            padx = 4,
            pady = 4
        )

        ttk.Entry(
            frame,
            textvariable = self.output_path_var
        ).grid(
            row = 1,
            column = 1,
            sticky = tk.EW,
            padx = 4,
            pady = 4
        )

        ttk.Button(
            frame,
            text = r'Browse Output',
            command = self._browse_output
        ).grid(
            row = 1,
            column = 2,
            sticky = tk.E,
            padx = 4,
            pady = 4
        )

        ttk.Checkbutton(
            frame,
            text = r'Video mode',
            variable = self.video_var
        ).grid(
            row = 2,
            column = 0,
            sticky = tk.W,
            padx = 4,
            pady = 4
        )

        ttk.Label(
            frame,
            text = r'Mode'
        ).grid(
            row = 2,
            column = 1,
            sticky = tk.W,
            padx = 4,
            pady = 4
        )

        ttk.Combobox(
            frame,
            values = [r'full', r'row', r'column', r'cluster'],
            state = r'readonly',
            textvariable = self.mode_var,
            width = 18
        ).grid(
            row = 2,
            column = 1,
            sticky = tk.E,
            padx = 4,
            pady = 4
        )

        ttk.Label(
            frame,
            text = r'Strip size'
        ).grid(
            row = 3,
            column = 0,
            sticky = tk.W,
            padx = 4,
            pady = 4
        )

        ttk.Spinbox(
            frame,
            from_ = 1,
            to = 4096,
            textvariable = self.strip_size_var,
            width = 12
        ).grid(
            row = 3,
            column = 0,
            sticky = tk.E,
            padx = 4,
            pady = 4
        )

        ttk.Label(
            frame,
            text = r'Cluster size'
        ).grid(
            row = 3,
            column = 1,
            sticky = tk.W,
            padx = 4,
            pady = 4
        )

        ttk.Spinbox(
            frame,
            from_ = 1,
            to = 4096,
            textvariable = self.cluster_size_var,
            width = 12
        ).grid(
            row = 3,
            column = 1,
            sticky = tk.E,
            padx = 4,
            pady = 4
        )

        ttk.Checkbutton(
            frame,
            text = r'Cluster reduced (no upscale)',
            variable = self.cluster_reduced_var
        ).grid(
            row = 4,
            column = 0,
            sticky = tk.W,
            padx = 4,
            pady = 4
        )

        ttk.Checkbutton(
            frame,
            text = r'Video reverse timeline',
            variable = self.video_reverse_var
        ).grid(
            row = 4,
            column = 1,
            sticky = tk.W,
            padx = 4,
            pady = 4
        )

        ttk.Button(
            frame,
            text = r'Run',
            command = self._run
        ).grid(
            row = 5,
            column = 0,
            sticky = tk.W,
            padx = 4,
            pady = 8
        )

        self.swatch_label = tk.Label(
            frame,
            text = r'    ',
            bg = r'#000000',
            relief = tk.GROOVE
        )
        self.swatch_label.grid(
            row = 5,
            column = 1,
            sticky = tk.W,
            padx = 4,
            pady = 8
        )

        ttk.Label(
            frame,
            textvariable = self.result_text_var
        ).grid(
            row = 5,
            column = 1,
            sticky = tk.E,
            padx = 4,
            pady = 8
        )

        self.preview_label = tk.Label(
            frame,
            bd = 1,
            relief = tk.SUNKEN,
            anchor = tk.CENTER
        )
        self.preview_label.grid(
            row = 6,
            column = 0,
            columnspan = 3,
            sticky = tk.NSEW,
            padx = 4,
            pady = 8
        )

        frame.rowconfigure(
            6,
            weight = 1
        )

    def _browse_input(
        self
    ) -> None:
        path = filedialog.askopenfilename()
        if path:
            self.input_path_var.set(
                path
            )

    def _browse_output(
        self
    ) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension = r'.png'
        )
        if path:
            self.output_path_var.set(
                path
            )

    def _run(
        self
    ) -> None:
        try:
            input_path = self.input_path_var.get().strip()
            output_path = self.output_path_var.get().strip()
            mode = self.mode_var.get()

            if input_path == r'':
                raise ValueError(
                    r'Input path is required.'
                )

            strip_size = self.strip_size_var.get()
            cluster_size = self.cluster_size_var.get()

            if self.video_var.get():
                self._run_video(
                    input_path = input_path,
                    output_path = output_path,
                    mode = mode,
                    strip_size = strip_size,
                    cluster_size = cluster_size
                )
            else:
                self._run_image(
                    input_path = input_path,
                    output_path = output_path,
                    mode = mode,
                    strip_size = strip_size,
                    cluster_size = cluster_size
                )

        except (ValueError, OSError) as exc:
            messagebox.showerror(
                title = r'MosaicMood Error',
                message = str(
                    exc
                )
            )
        except Exception as exc:
            messagebox.showerror(
                title = r'MosaicMood Error',
                message = (
                    r'Unexpected internal error. '
                    r'Please try again and check your input settings. '
                    f"Details: {exc}"
                )
            )

    def _run_image(
        self,
        input_path: str,
        output_path: str,
        mode: str,
        strip_size: int | None = None,
        cluster_size: int | None = None
    ) -> None:
        resolved_strip_size = (
            self.strip_size_var.get()
            if strip_size is None
            else strip_size
        )
        resolved_cluster_size = (
            self.cluster_size_var.get()
            if cluster_size is None
            else cluster_size
        )

        result = process_image_request(
            input_path = input_path,
            mode = mode,
            strip_size = resolved_strip_size,
            cluster_size = resolved_cluster_size,
            cluster_reduced = self.cluster_reduced_var.get(),
            show_progress = False
        )

        if result.mode == r'full':
            assert result.average_rgb is not None
            rgb = result.average_rgb
            self._set_swatch(
                rgb
            )
            self.result_text_var.set(
                f"Result: average (R, G, B) = {rgb}"
            )

            preview = self._solid_preview(
                rgb = rgb,
                width = 320,
                height = 120
            )
            self._set_preview_image(
                preview
            )
            return

        assert result.result_img is not None
        result_img = result.result_img

        if output_path != r'':
            save_rgb_image(
                output_path,
                result_img
            )

        self._set_preview_image(
            result_img
        )

        self.result_text_var.set(
            f"Result: {result.mode} image size {result_img.shape[1]}x{result_img.shape[0]}"
        )

    def _run_video(
        self,
        input_path: str,
        output_path: str,
        mode: str,
        strip_size: int | None = None,
        cluster_size: int | None = None
    ) -> None:
        resolved_strip_size = (
            self.strip_size_var.get()
            if strip_size is None
            else strip_size
        )
        resolved_cluster_size = (
            self.cluster_size_var.get()
            if cluster_size is None
            else cluster_size
        )

        result = process_video_request(
            input_path = input_path,
            mode = mode,
            strip_size = resolved_strip_size,
            cluster_size = resolved_cluster_size,
            cluster_reduced = self.cluster_reduced_var.get(),
            reverse = self.video_reverse_var.get(),
            show_progress = False,
            progress_desc = r'Video timeline'
        )

        timeline_img = result.timeline_img
        frame_count = result.frame_count

        if output_path != r'':
            save_rgb_image(
                output_path,
                timeline_img
            )

        preview = self._stretch_timeline_for_preview(
            timeline_img
        )
        self._set_preview_image(
            preview
        )

        self.result_text_var.set(
            f"Result: video timeline {timeline_img.shape[1]}x{timeline_img.shape[0]}, frames {frame_count}"
        )

    def _set_swatch(
        self,
        rgb: RGBTuple
    ) -> None:
        r, g, b = rgb
        hex_colour = f"#{r:02x}{g:02x}{b:02x}"
        self.swatch_label.configure(
            bg = hex_colour
        )

    def _set_preview_image(
        self,
        img_rgb: ImageArray
    ) -> None:
        preview = self._fit_for_preview(
            img_rgb = img_rgb,
            max_width = 920,
            max_height = 420
        )

        bgr: ImageArray = cast(
            ImageArray,
            cv2.cvtColor(
                preview,
                cv2.COLOR_RGB2BGR
            )
        )

        ok, encoded = cv2.imencode(
            r'.png',
            bgr
        )
        if not ok:
            raise ValueError(
                r'Failed to encode preview image.'
            )

        encoded_array: npt.NDArray[np.uint8] = cast(
            npt.NDArray[np.uint8],
            encoded
        )
        data_bytes: bytes = encoded_array.tobytes()
        data: bytes = base64.b64encode(
            data_bytes
        )

        self.preview_image = tk.PhotoImage(
            data = data
        )
        assert self.preview_image is not None
        self.preview_label.configure(
            image = self.preview_image,
            text = r''
        )

    def _fit_for_preview(
        self,
        img_rgb: ImageArray,
        max_width: int,
        max_height: int
    ) -> ImageArray:
        return fit_image_nearest(
            img_rgb,
            max_width = max_width,
            max_height = max_height
        )

    def _solid_preview(
        self,
        rgb: RGBTuple,
        width: int,
        height: int
    ) -> ImageArray:
        r, g, b = rgb
        preview = np.full(
            (height, width, 3),
            (r, g, b),
            dtype = np.uint8
        )

        return preview

    def _stretch_timeline_for_preview(
        self,
        timeline_img: ImageArray
    ) -> ImageArray:
        target_height = 96

        return stretch_timeline_nearest(
            timeline_img,
            target_height = target_height
        )


def main() -> None:
    root = tk.Tk()
    MosaicMoodApp(
        root
    )
    root.mainloop()


if __name__ == r'__main__':
    main()
