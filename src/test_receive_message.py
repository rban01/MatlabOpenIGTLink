"""
test_receive_message.py
Replaces: testReceiveMessage.m

Connects to a running 3D Slicer OpenIGTLinkIF server, waits for N messages
(skipping the initial STATUS handshake), then disconnects.
"""
import numpy as np

from igtl_connect import igtl_connect, igtl_disconnect
from OpenIGTLinkMessageReceiver import OpenIGTLinkMessageReceiver
from igtl_show_image import igtl_show_image


# -----------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------

def on_rx_status_message(device_name: str, text: str) -> None:
    print(f"Received STATUS message: {device_name}  {text}")


def on_rx_string_message(device_name: str, text: str) -> None:
    print(f"Received STRING message: {device_name} = {text}")


def on_rx_transform_message(device_name: str, transform: np.ndarray) -> None:
    print("Received TRANSFORM message:")
    print(f"  {device_name} =")
    print(transform)


def on_rx_point_message(device_name: str, point_list: np.ndarray) -> None:
    print("Received POINT message:")
    print(f"  {device_name} =")
    print(point_list)


def on_rx_image_message(device_name: str, image: dict) -> None:
    print("Received IMAGE message:")
    print(f"  {device_name}")
    print(f"  Origin  = {image.get('origin',  'N/A')}")
    print(f"  Spacing = {image.get('spacing', 'N/A')}")
    igtl_show_image(image, slice_idx=1)
    np.save('RTDose.npy', image['matrix'])


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def test_receive_message() -> None:
    N = 1   # number of data messages to receive (not counting initial STATUS)

    client = igtl_connect('127.0.0.1', 18944)

    receiver = OpenIGTLinkMessageReceiver(
        client,
        on_status    = on_rx_status_message,
        on_string    = on_rx_string_message,
        on_transform = on_rx_transform_message,
        on_point     = on_rx_point_message,
        on_image     = on_rx_image_message,
        timeout      = 5.0,
    )

    for _ in range(N + 1):    # +1 to consume the initial STATUS message
        receiver.read_message()

    igtl_disconnect(client)


if __name__ == '__main__':
    test_receive_message()
