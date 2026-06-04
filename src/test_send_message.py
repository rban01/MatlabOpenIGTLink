"""
test_send_message.py
Replaces: testSendMessage.m

Sends STRING, TRANSFORM, POINT, and IMAGE messages to a running 3D Slicer
OpenIGTLinkIF server, then disconnects.

IMAGE 1: loaded from 'RTDose.npy' (saved by test_receive_message.py).
IMAGE 2: programmatically generated uint16 gradient volume.
"""
import numpy as np

from igtl_connect import igtl_connect, igtl_disconnect
from OpenIGTLinkMessageSender import OpenIGTLinkMessageSender
from igtl_show_image import igtl_show_image


# -----------------------------------------------------------------------
# Utility: Convert LPS (NumPy default) -> RAS (3D Slicer standard)
# -----------------------------------------------------------------------

def convert_to_ras(image_lps: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees then flip left-right to match 3D Slicer's RAS convention."""
    image_ras = np.rot90(image_lps)
    image_ras = np.fliplr(image_ras)
    return image_ras


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def test_send_message() -> None:

    # Connect
    client = igtl_connect('127.0.0.1', 18944)
    sender = OpenIGTLinkMessageSender(client)

    # STRING message
    sender.send_string('StringTest', 'Hello World!')

    # TRANSFORM message
    theta = 0.0
    tx, ty, tz = 1.0, 2.0, 3.0
    matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, tx],
        [np.sin(theta),  np.cos(theta), 0.0, ty],
        [0.0,            0.0,           1.0, tz],
        [0.0,            0.0,           0.0, 1.0],
    ], dtype=np.float32)
    sender.send_transform('TransformTest', matrix)

    # POINT messages
    point_list_f = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)
    sender.send_point('F', point_list_f)

    point_list_p = np.array([
        [-20, -15, -10],
        [-20, -15,  10],
        [-20,  15, -10],
        [-20,  15,  10],
        [ 20, -15, -10],
        [ 20, -15,  10],
        [ 20,  15, -10],
        [ 20,  15,  10],
    ], dtype=np.float32)
    sender.send_point('P', point_list_p)

    # IMAGE 1 - RTDose example (loaded from test_receive_message output)
    try:
        dose_matrix = np.load('RTDose.npy')
        dose_image = {'matrix': dose_matrix, 'coordinate': 1}   # assumed RAS
        igtl_show_image(dose_image, slice_idx=1)
        sender.send_image('IMAGE_1', dose_image)
    except FileNotFoundError:
        print("RTDose.npy not found; skipping IMAGE_1. "
              "Run test_receive_message.py first to generate it.")

    # IMAGE 2 - Programmatically generated uint16 gradient volume
    width, height, depth = 320, 240, 5
    N = height // 4

    base_gradient = np.tile(
        np.linspace(0, 65535, width, dtype=np.uint16)[np.newaxis, :],
        (height, 1),
    )
    matrix_lps = np.stack([base_gradient] * depth, axis=2)

    for d in range(depth):
        shade = int(65535 * (1.0 - d / max(depth - 1, 1)))
        matrix_lps[:N, :N, d] = shade

    image_lps = {'matrix': matrix_lps, 'coordinate': 2}   # LPS

    matrix_ras = convert_to_ras(matrix_lps)
    image_ras  = {'matrix': matrix_ras, 'coordinate': 1}  # RAS

    igtl_show_image(image_ras, slice_idx=1)
    sender.send_image('IMAGE_2', image_ras)

    # Disconnect
    igtl_disconnect(client)


if __name__ == '__main__':
    test_send_message()
