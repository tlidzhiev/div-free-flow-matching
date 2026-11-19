import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert an image of shape (H, W), (H, W, C) or (C, H, W) with 1 or 3 channels, either
    float in [0, 1] or uint8, to a uint8 RGB array of shape (H, W, 3).
    """
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = img.transpose(1, 2, 0)
    if img.ndim == 2:
        img = img[..., None]
    if img.dtype != np.uint8:
        img = (np.clip(img.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.broadcast_to(img, (*img.shape[:2], 3))


def caption_image(
    img: np.ndarray,
    text: str,
    color: str = 'black',
    min_size: int = 96,
) -> np.ndarray:
    """
    Add a caption under an image.

    Parameters
    ----------
    img : np.ndarray
        Image of shape (H, W), (H, W, C) or (C, H, W), float in [0, 1] or uint8.
    text : str
        Caption.
    color : str, optional
        Color of the caption.
    min_size : int, optional
        Small images are upscaled (by an integer factor) to at least this width, so the
        caption fits.

    Returns
    -------
    np.ndarray
        uint8 RGB image of shape (H', W', 3).
    """
    rgb = _to_uint8_rgb(img)
    scale = max(1, -(-min_size // rgb.shape[1]))
    rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)

    font = ImageFont.load_default()
    left, top, right, bottom = (int(x) for x in font.getbbox(text))
    text_width, text_height = right - left, bottom - top
    height, width = rgb.shape[:2]

    canvas = Image.new('RGB', (max(width, text_width + 4), height + text_height + 6), 'white')
    canvas.paste(Image.fromarray(rgb), ((canvas.width - width) // 2, 0))
    ImageDraw.Draw(canvas).text(
        ((canvas.width - text_width) // 2 - left, height + 3 - top), text, fill=color, font=font
    )
    return np.asarray(canvas)


def make_grid(
    images: list[np.ndarray],
    nrow: int = 8,
    padding: int = 2,
    pad_value: int = 0,
) -> np.ndarray:
    """
    Make a grid of images of the same shape.

    Parameters
    ----------
    images : list[np.ndarray]
        uint8 images of shape (H, W, C).
    nrow : int, optional
        Number of images in a row of the grid.
    padding : int, optional
        Space between the images.
    pad_value : int, optional
        Value of the padding pixels.

    Returns
    -------
    np.ndarray
        uint8 image of shape (H', W', C).
    """
    num_cols = min(nrow, len(images))
    num_rows = -(-len(images) // num_cols)
    height, width, channels = images[0].shape
    grid = np.full(
        (
            num_rows * (height + padding) + padding,
            num_cols * (width + padding) + padding,
            channels,
        ),
        pad_value,
        dtype=np.uint8,
    )
    for i, image in enumerate(images):
        y = padding + (i // num_cols) * (height + padding)
        x = padding + (i % num_cols) * (width + padding)
        grid[y : y + height, x : x + width] = image
    return grid
