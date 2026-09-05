# Mosaic Mood

MosaicMood is a Python tool that extracts colour summaries from images and videos.

This has been made with GitHub copilot, hence why it is uploaded here.

It supports: 

- Full-image average colour
- Row average strip image
- Column average strip image
- Clustered mosaic image (reduced or upscaled)
- Video timeline image where each pixel column represents one frame average
- CLI and Tk GUI usage

## Requirements

- Python 3.12 recommended
- OpenCV

Detailed dependencies can be found in `requirements.txt`.

## Quick Start (CLI)

General form:

```sh
python main.py INPUT_PATH [options]
```

Module entrypoint form:

```sh
python -m mosaicmood INPUT_PATH [options]
```

### Image Modes

1. Full average colour (terminal output only):

This returns one colour, which is the average colour of the entire image.

```sh
python main.py INPUT_PATH --mode full
```

2. Row average strip (strict default width = 1 pixel):

This returns a column where each pixel is the average colour of each row.

```sh
python main.py INPUT_PATH -- mode row --output OUTPUT_PATH
```

3. Column average strip (strict default height = 1 pixel):

This returns a row where each pixel is the average colour of each row.

```sh
python main.py INPUT_PATH --mode column --ouput OUTPUT_PATH
```

4. Cluster mosaic (default is upscaled back to source dimensions):

This turns the image to a mosaic. The `--cluster-size` is the size of the mosaic boxes. If a small value is used for the cluster size, it might be a little more computationally intensive.

By default, the result image has the resolution of the input image, otherwise the result image will be the original size divided by the cluster size.

```sh
python main.py INPUT_PATH --mode cluster --cluster-size 10 --output OUTPUT_PATH
```

For the reduced result image use:

```sh
python main.py INPUT_PATH --mode cluster --cluster-size 10 --cluster-reduced --output OUTPUT_PATH
```

5. Preview in terminal only (no output file):

A funny little functionality, or useful if there is no graphical environment. True-colour terminal is required.

```sh
python main.py INPUT_PATH -- mode cluster
```

### Video Mode

Video mode creates a timeline image (default height = 1) where each column is the average colour of one processed frame.

1. Full-frame timeline:

```sh
python main.py INPUT_PATH --video --mode full --output OUTPUT_PATH
```

2. Reverse timeline order (last frame on the left):

```sh
python main.py INPUT_PATH --video --mode full --video-reverse --output OUTPUT_PATH
```

3. Reuse row/column/cluster processing per frame before averaging:

```sh
python main.py INPUT_PATH --video --mode cluster --cluster-size 8 --output OUTPUT_PATH
```

4. Video preview in terminal only (no output file):

```sh
python main.py INPUT_PATH --video --mode full
```

## Flags

- `--mode {full,row,column,cluster}`:
  Selects processing mode.
- `--output PATH`:
  Saves image output to `PATH`. If omitted, a terminal preview is shown.
- `--strip-size N`:
  Width for row mode or height for column mode. Default is `1`.

  This just stretches the image to `N` pixels, to make the resulting image easier to see.
- `--cluster-size N`:
  Cluster block size for cluster mode. Default is `10`.
- `--cluster-reduced`:
  Keeps cluster result at reduced size intead of upscaling to source dimensions.
- `--video`:
  Enables video timeline processing.
- `--video-reverse`:
  Reverses frame order for the timeline image.

## GUI Usage

The application also offers a Tk GUI, thinking it comes with Python by default. It turns out you might need to install `python3-tk` on Ubuntu/Debian and, I imagine, `tk` with `pip` elsewhere, if this doesn't work.

```bash
python gui.py
```

If no output path is provided in the GUI, a preview is shown inside the application window.
