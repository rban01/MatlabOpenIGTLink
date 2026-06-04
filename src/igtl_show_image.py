"""
igtl_show_image.py
Replaces: igtlShowImage.m

Displays one slice of a 2-D/3-D/4-D OpenIGTLink image using matplotlib.
Accepts either a raw numpy array or the image dict produced by
OpenIGTLinkMessageReceiver.

Requires: numpy, matplotlib
"""
from __future__ import annotations
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt


def igtl_show_image(
    image_input: Union[np.ndarray, dict],
    slice_idx: Optional[int] = None,
) -> None:
    """
    Display one slice of an OpenIGTLink image.

    Parameters
    ----------
    image_input : ndarray or dict
        ndarray  - 2-D, 3-D, or 4-D (last axis = channels).
        dict with keys:
            'matrix'     - numpy array  (required)
            'coordinate' - 1=RAS, 2=LPS (optional, default 2)
    slice_idx : int or None
        1-based slice number.
        None or omitted -> middle slice.
        <= 0            -> last slice.
    """
    if isinstance(image_input, np.ndarray):
        matrix     = image_input
        coordinate = 2
    elif isinstance(image_input, dict):
        matrix     = np.asarray(image_input['matrix'])
        coordinate = image_input.get('coordinate', 2)
    else:
        raise TypeError("image_input must be a numpy array or dict with 'matrix'.")

    if matrix.ndim < 2 or matrix.ndim > 4:
        raise ValueError("Image matrix must be 2-D, 3-D, or 4-D.")

    if matrix.ndim == 2:
        matrix = matrix[:, :, np.newaxis]

    if matrix.ndim == 3:
        width, height, depth = matrix.shape
        channels = 1
    else:
        width, height, depth, channels = matrix.shape

    if slice_idx is None:
        slice_idx = max(1, (depth + 1) // 2)
    elif slice_idx <= 0:
        slice_idx = depth
    elif slice_idx > depth:
        raise ValueError(f"slice_idx {slice_idx} exceeds depth {depth}.")

    print(
        f"Image Dimensions: Width={width}, Height={height}, "
        f"Depth={depth}, Channels={channels}"
    )

    if depth > 1:
        img = matrix[:, :, slice_idx - 1]
    else:
        img = matrix[:, :, 0]

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    if coordinate == 1:
        img = np.rot90(img)
        img = np.fliplr(img)
        coord_label = 'RAS (displayed as LPS)'
    elif coordinate == 2:
        coord_label = 'LPS'
    else:
        raise ValueError(f"Invalid coordinate: {coordinate}. Use 1=RAS or 2=LPS.")

    plt.figure()
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None, aspect='equal')
    plt.title(f"Slice {slice_idx}/{depth}  ({coord_label})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
