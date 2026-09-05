import numpy as np
import pytest
import runpy
import sys
import types

import gui
from gui import MosaicMoodApp


class DummyLabel:
    def __init__(
        self
    ):
        self.values = {}

    def configure(
        self,
        **kwargs
    ):
        self.values.update(
            kwargs
        )


class DummyVar:
    def __init__(
        self,
        value = None
    ):
        self.value = value

    def get(
        self
    ):
        return self.value

    def set(
        self,
        value
    ):
        self.value = value


class FakeWidget:
    def __init__(
        self,
        *_args,
        **_kwargs
    ):
        self.values = {}

    def grid(
        self,
        **_kwargs
    ):
        return self

    def pack(
        self,
        **_kwargs
    ):
        return self

    def columnconfigure(
        self,
        *_args,
        **_kwargs
    ):
        return None

    def rowconfigure(
        self,
        *_args,
        **_kwargs
    ):
        return None

    def configure(
        self,
        **kwargs
    ):
        self.values.update(
            kwargs
        )


class FakeRoot(FakeWidget):
    def __init__(
        self
    ):
        super().__init__()
        self.title_value = None
        self.geometry_value = None
        self.mainloop_called = 0

    def title(
        self,
        value
    ):
        self.title_value = value

    def geometry(
        self,
        value
    ):
        self.geometry_value = value

    def mainloop(
        self
    ):
        self.mainloop_called += 1


class FakePhoto:
    def __init__(
        self,
        data = None
    ):
        self.data = data


def test_solid_preview_uses_requested_colour() -> None:
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    preview = app._solid_preview(
        rgb = (12, 34, 56),
        width = 5,
        height = 3
    )

    assert preview.shape == (3, 5, 3)
    assert np.all(
        preview[:, :, 0] == 12
    )
    assert np.all(
        preview[:, :, 1] == 34
    )
    assert np.all(
        preview[:, :, 2] == 56
    )


def test_init_builds_ui_with_mocked_tk(
    monkeypatch
):
    root = FakeRoot()

    monkeypatch.setattr(
        gui.tk,
        r'StringVar',
        lambda value = r'': DummyVar(value)
    )
    monkeypatch.setattr(
        gui.tk,
        r'BooleanVar',
        lambda value = False: DummyVar(value)
    )
    monkeypatch.setattr(
        gui.tk,
        r'IntVar',
        lambda value = 0: DummyVar(value)
    )

    monkeypatch.setattr(gui.tk, r'BOTH', 0)
    monkeypatch.setattr(gui.tk, r'W', 0)
    monkeypatch.setattr(gui.tk, r'E', 0)
    monkeypatch.setattr(gui.tk, r'EW', 0)
    monkeypatch.setattr(gui.tk, r'NSEW', 0)
    monkeypatch.setattr(gui.tk, r'CENTER', 0)
    monkeypatch.setattr(gui.tk, r'GROOVE', 0)
    monkeypatch.setattr(gui.tk, r'SUNKEN', 0)

    monkeypatch.setattr(gui.tk, r'Label', FakeWidget)

    monkeypatch.setattr(gui.ttk, r'Frame', FakeWidget)
    monkeypatch.setattr(gui.ttk, r'Label', FakeWidget)
    monkeypatch.setattr(gui.ttk, r'Entry', FakeWidget)
    monkeypatch.setattr(gui.ttk, r'Button', FakeWidget)
    monkeypatch.setattr(gui.ttk, r'Checkbutton', FakeWidget)
    monkeypatch.setattr(gui.ttk, r'Combobox', FakeWidget)
    monkeypatch.setattr(gui.ttk, r'Spinbox', FakeWidget)

    app = MosaicMoodApp(
        root
    )

    assert root.title_value == r'MosaicMood'
    assert root.geometry_value == r'980x640'
    assert app.swatch_label is not None
    assert app.preview_label is not None


def test_fit_for_preview_limits_dimensions() -> None:
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    img = np.zeros(
        (200, 400, 3),
        dtype = np.uint8
    )

    resized = app._fit_for_preview(
        img_rgb = img,
        max_width = 100,
        max_height = 60
    )

    assert resized.shape[1] <= 100
    assert resized.shape[0] <= 60


def test_stretch_timeline_for_preview_sets_fixed_height() -> None:
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    timeline = np.zeros(
        (1, 7, 3),
        dtype = np.uint8
    )

    stretched = app._stretch_timeline_for_preview(
        timeline
    )

    assert stretched.shape == (96, 7, 3)


def test_set_swatch_updates_hex_colour() -> None:
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )
    app.swatch_label = DummyLabel()

    app._set_swatch(
        (16, 32, 48)
    )

    assert app.swatch_label.values[r'bg'] == r'#102030'


def test_run_requires_input_and_shows_error(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.input_path_var = DummyVar(
        r''
    )
    app.output_path_var = DummyVar(
        r''
    )
    app.mode_var = DummyVar(
        r'full'
    )
    app.video_var = DummyVar(
        False
    )

    called = {
        r'error': 0
    }

    def fake_showerror(
        title,
        message
    ):
        called[r'error'] += 1
        assert title == r'MosaicMood Error'
        assert r'Input path is required.' in message

    monkeypatch.setattr(
        gui.messagebox,
        r'showerror',
        fake_showerror
    )

    app._run()

    assert called[r'error'] == 1


def test_run_dispatches_video_branch() -> None:
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.input_path_var = DummyVar(
        r'input.avi'
    )
    app.output_path_var = DummyVar(
        r'out.png'
    )
    app.mode_var = DummyVar(
        r'full'
    )
    app.video_var = DummyVar(
        True
    )
    app.strip_size_var = DummyVar(1)
    app.cluster_size_var = DummyVar(10)

    calls = {
        r'video': 0
    }

    def fake_video(
        **kwargs
    ):
        calls[r'video'] += 1
        assert kwargs[r'input_path'] == r'input.avi'

    app._run_video = fake_video
    app._run_image = lambda **_kwargs: None

    app._run()

    assert calls[r'video'] == 1


def test_run_dispatches_image_branch() -> None:
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.input_path_var = DummyVar(
        r'input.png'
    )
    app.output_path_var = DummyVar(
        r''
    )
    app.mode_var = DummyVar(
        r'row'
    )
    app.video_var = DummyVar(
        False
    )
    app.strip_size_var = DummyVar(1)
    app.cluster_size_var = DummyVar(10)

    calls = {
        r'image': 0
    }

    app._run_video = lambda **_kwargs: None

    def fake_image(
        **kwargs
    ):
        calls[r'image'] += 1
        assert kwargs[r'input_path'] == r'input.png'

    app._run_image = fake_image

    app._run()

    assert calls[r'image'] == 1


def test_run_image_full_updates_text_and_preview(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.result_text_var = DummyVar(
        r''
    )
    app.strip_size_var = DummyVar(1)
    app.cluster_size_var = DummyVar(10)
    app.cluster_reduced_var = DummyVar(False)

    calls = {
        r'swatch': 0,
        r'preview': 0
    }

    app._set_swatch = lambda _rgb: calls.__setitem__(r'swatch', calls[r'swatch'] + 1)
    app._set_preview_image = lambda _img: calls.__setitem__(r'preview', calls[r'preview'] + 1)

    monkeypatch.setattr(
        gui,
        r'process_image_request',
        lambda **_kwargs: type(
            r'Result',
            (),
            {
                r'mode': r'full',
                r'source_img': np.zeros((4, 6, 3), dtype = np.uint8),
                r'average_rgb': (1, 2, 3),
                r'result_img': None,
            }
        )()
    )

    app._run_image(
        input_path = r'in.png',
        output_path = r'',
        mode = r'full'
    )

    assert calls[r'swatch'] == 1
    assert calls[r'preview'] == 1
    assert r'average (R, G, B) = (1, 2, 3)' in app.result_text_var.get()


def test_run_video_updates_text_and_saves(
    monkeypatch,
    tmp_path
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.strip_size_var = DummyVar(1)
    app.cluster_size_var = DummyVar(3)
    app.cluster_reduced_var = DummyVar(False)
    app.video_reverse_var = DummyVar(False)
    app.result_text_var = DummyVar(r'')

    app._stretch_timeline_for_preview = lambda img: img
    app._set_preview_image = lambda _img: None

    monkeypatch.setattr(
        gui,
        r'process_video_request',
        lambda **_kwargs: type(
            r'VideoResult',
            (),
            {
                r'timeline_img': np.zeros((1, 4, 3), dtype = np.uint8),
                r'frame_count': 4,
            }
        )()
    )

    saved = {
        r'count': 0
    }

    def fake_save(
        _path,
        _img
    ):
        saved[r'count'] += 1

    monkeypatch.setattr(
        gui,
        r'save_rgb_image',
        fake_save
    )

    app._run_video(
        input_path = r'video.avi',
        output_path = str(tmp_path / r'timeline.png'),
        mode = r'full'
    )

    assert saved[r'count'] == 1
    assert r'frames 4' in app.result_text_var.get()


def test_run_image_row_saves_and_sets_preview(
    monkeypatch,
    tmp_path
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.strip_size_var = DummyVar(2)
    app.cluster_size_var = DummyVar(3)
    app.cluster_reduced_var = DummyVar(False)
    app.result_text_var = DummyVar(r'')

    app._set_preview_image = lambda _img: None

    monkeypatch.setattr(
        gui,
        r'process_image_request',
        lambda **_kwargs: type(
            r'Result',
            (),
            {
                r'mode': r'row',
                r'source_img': np.zeros((4, 6, 3), dtype = np.uint8),
                r'average_rgb': None,
                r'result_img': np.zeros((4, 2, 3), dtype = np.uint8),
            }
        )()
    )

    saved = {
        r'count': 0
    }

    def fake_save(
        _path,
        _img
    ):
        saved[r'count'] += 1

    monkeypatch.setattr(
        gui,
        r'save_rgb_image',
        fake_save
    )

    app._run_image(
        input_path = r'input.png',
        output_path = str(tmp_path / r'row.png'),
        mode = r'row'
    )

    assert saved[r'count'] == 1
    assert r'row image size' in app.result_text_var.get()


def test_run_image_column_branch(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.strip_size_var = DummyVar(2)
    app.cluster_size_var = DummyVar(3)
    app.cluster_reduced_var = DummyVar(False)
    app.result_text_var = DummyVar(r'')
    app._set_preview_image = lambda _img: None

    captured = {
        r'strip_size': None
    }

    def fake_process_image_request(
        **kwargs
    ):
        captured[r'strip_size'] = kwargs[r'strip_size']
        return type(
            r'Result',
            (),
            {
                r'mode': r'column',
                r'source_img': np.zeros((4, 6, 3), dtype = np.uint8),
                r'average_rgb': None,
                r'result_img': np.zeros((2, 6, 3), dtype = np.uint8),
            }
        )()

    monkeypatch.setattr(
        gui,
        r'process_image_request',
        fake_process_image_request
    )

    app._run_image(
        input_path = r'input.png',
        output_path = r'',
        mode = r'column'
    )

    assert captured[r'strip_size'] == 2
    assert r'column image size' in app.result_text_var.get()


def test_run_image_cluster_calls_upscale_when_not_reduced(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.strip_size_var = DummyVar(1)
    app.cluster_size_var = DummyVar(3)
    app.cluster_reduced_var = DummyVar(False)
    app.result_text_var = DummyVar(r'')

    app._set_preview_image = lambda _img: None

    calls = {
        r'cluster_reduced': None
    }

    def fake_process_image_request(
        **kwargs
    ):
        calls[r'cluster_reduced'] = kwargs[r'cluster_reduced']
        return type(
            r'Result',
            (),
            {
                r'mode': r'cluster',
                r'source_img': np.zeros((4, 6, 3), dtype = np.uint8),
                r'average_rgb': None,
                r'result_img': np.zeros((4, 6, 3), dtype = np.uint8),
            }
        )()

    monkeypatch.setattr(
        gui,
        r'process_image_request',
        fake_process_image_request
    )

    app._run_image(
        input_path = r'input.png',
        output_path = r'',
        mode = r'cluster'
    )

    assert calls[r'cluster_reduced'] is False


def test_run_image_cluster_skips_upscale_when_reduced(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.strip_size_var = DummyVar(1)
    app.cluster_size_var = DummyVar(3)
    app.cluster_reduced_var = DummyVar(True)
    app.result_text_var = DummyVar(r'')

    app._set_preview_image = lambda _img: None

    calls = {
        r'cluster_reduced': None
    }

    def fake_process_image_request(
        **kwargs
    ):
        calls[r'cluster_reduced'] = kwargs[r'cluster_reduced']
        return type(
            r'Result',
            (),
            {
                r'mode': r'cluster',
                r'source_img': np.zeros((4, 6, 3), dtype = np.uint8),
                r'average_rgb': None,
                r'result_img': np.zeros((2, 3, 3), dtype = np.uint8),
            }
        )()

    monkeypatch.setattr(
        gui,
        r'process_image_request',
        fake_process_image_request
    )

    app._run_image(
        input_path = r'input.png',
        output_path = r'',
        mode = r'cluster'
    )

    assert calls[r'cluster_reduced'] is True


def test_browse_input_sets_variable(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )
    app.input_path_var = DummyVar(r'')

    monkeypatch.setattr(
        gui.filedialog,
        r'askopenfilename',
        lambda: r'/tmp/in.png'
    )

    app._browse_input()

    assert app.input_path_var.get() == r'/tmp/in.png'


def test_browse_output_sets_variable(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )
    app.output_path_var = DummyVar(r'')

    monkeypatch.setattr(
        gui.filedialog,
        r'asksaveasfilename',
        lambda **_kwargs: r'/tmp/out.png'
    )

    app._browse_output()

    assert app.output_path_var.get() == r'/tmp/out.png'


def test_set_preview_image_sets_photo_on_label(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.preview_label = DummyLabel()

    monkeypatch.setattr(
        app,
        r'_fit_for_preview',
        lambda img_rgb, max_width, max_height: img_rgb
    )

    class DummyPhoto:
        def __init__(
            self,
            data = None
        ):
            self.data = data

    monkeypatch.setattr(
        gui.tk,
        r'PhotoImage',
        DummyPhoto
    )

    img = np.zeros(
        (4, 5, 3),
        dtype = np.uint8
    )

    app._set_preview_image(
        img
    )

    assert isinstance(
        app.preview_image,
        DummyPhoto
    )
    assert app.preview_label.values[r'text'] == r''


def test_set_preview_image_raises_when_encoding_fails(
    monkeypatch
):
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.preview_label = DummyLabel()

    monkeypatch.setattr(
        app,
        r'_fit_for_preview',
        lambda img_rgb, max_width, max_height: img_rgb
    )

    monkeypatch.setattr(
        gui.cv2,
        r'imencode',
        lambda _ext, _img: (False, None)
    )

    with pytest.raises(
        ValueError,
        match = r'Failed to encode preview image.'
    ):
        app._set_preview_image(
            np.zeros((2, 2, 3), dtype = np.uint8)
        )


def test_gui_main_calls_root_mainloop(
    monkeypatch
):
    root = FakeRoot()

    monkeypatch.setattr(
        gui.tk,
        r'Tk',
        lambda: root
    )

    monkeypatch.setattr(
        gui,
        r'MosaicMoodApp',
        lambda _root: None
    )

    gui.main()

    assert root.mainloop_called == 1


def test_gui_module_entrypoint_runs(
    monkeypatch,
    repo_root
):
    tk_module = types.ModuleType(
        r'tkinter'
    )
    ttk_module = types.ModuleType(
        r'tkinter.ttk'
    )
    filedialog_module = types.ModuleType(
        r'tkinter.filedialog'
    )
    messagebox_module = types.ModuleType(
        r'tkinter.messagebox'
    )

    tk_module.BOTH = 0
    tk_module.W = 0
    tk_module.E = 0
    tk_module.EW = 0
    tk_module.NSEW = 0
    tk_module.CENTER = 0
    tk_module.GROOVE = 0
    tk_module.SUNKEN = 0
    tk_module.StringVar = lambda value = r'': DummyVar(value)
    tk_module.BooleanVar = lambda value = False: DummyVar(value)
    tk_module.IntVar = lambda value = 0: DummyVar(value)
    tk_module.Label = FakeWidget
    tk_module.PhotoImage = FakePhoto

    root = FakeRoot()
    tk_module.Tk = lambda: root

    ttk_module.Frame = FakeWidget
    ttk_module.Label = FakeWidget
    ttk_module.Entry = FakeWidget
    ttk_module.Button = FakeWidget
    ttk_module.Checkbutton = FakeWidget
    ttk_module.Combobox = FakeWidget
    ttk_module.Spinbox = FakeWidget

    filedialog_module.askopenfilename = lambda: r''
    filedialog_module.asksaveasfilename = lambda **_kwargs: r''
    messagebox_module.showerror = lambda **_kwargs: None

    tk_module.ttk = ttk_module
    tk_module.filedialog = filedialog_module
    tk_module.messagebox = messagebox_module

    monkeypatch.setitem(sys.modules, r'tkinter', tk_module)
    monkeypatch.setitem(sys.modules, r'tkinter.ttk', ttk_module)
    monkeypatch.setitem(sys.modules, r'tkinter.filedialog', filedialog_module)
    monkeypatch.setitem(sys.modules, r'tkinter.messagebox', messagebox_module)

    runpy.run_path(
        str(repo_root / r'gui.py'),
        run_name = r'__main__'
    )

    assert root.mainloop_called == 1


def test_run_image_full_integration_uses_real_service(
    sample_image_path
) -> None:
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.strip_size_var = DummyVar(1)
    app.cluster_size_var = DummyVar(10)
    app.cluster_reduced_var = DummyVar(False)
    app.result_text_var = DummyVar(r'')

    calls = {
        r'swatch': 0,
        r'preview': 0
    }

    app._set_swatch = lambda _rgb: calls.__setitem__(r'swatch', calls[r'swatch'] + 1)
    app._set_preview_image = lambda _img: calls.__setitem__(r'preview', calls[r'preview'] + 1)

    app._run_image(
        input_path = str(sample_image_path),
        output_path = r'',
        mode = r'full'
    )

    assert calls[r'swatch'] == 1
    assert calls[r'preview'] == 1
    assert r'Result: average (R, G, B) =' in app.result_text_var.get()


def test_run_video_integration_uses_real_service(
    sample_video_path
) -> None:
    app = MosaicMoodApp.__new__(
        MosaicMoodApp
    )

    app.strip_size_var = DummyVar(1)
    app.cluster_size_var = DummyVar(10)
    app.cluster_reduced_var = DummyVar(False)
    app.video_reverse_var = DummyVar(False)
    app.result_text_var = DummyVar(r'')
    app._set_preview_image = lambda _img: None

    app._run_video(
        input_path = str(sample_video_path),
        output_path = r'',
        mode = r'full'
    )

    assert r'Result: video timeline' in app.result_text_var.get()
    assert r'frames 4' in app.result_text_var.get()
