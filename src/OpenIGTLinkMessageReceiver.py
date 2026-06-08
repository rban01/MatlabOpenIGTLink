"""
OpenIGTLinkMessageReceiver.py
Replaces: OpenIGTLinkMessageReceiver.m

Receives STATUS, STRING, TRANSFORM, POINT, and IMAGE messages over
OpenIGTLink and dispatches them to user-supplied callbacks.
pyigtl handles header parsing, body reading, and CRC64 verification
(igtlComputeCrc.m is therefore not required in Python).

Requires: pyigtl >= 0.2.0, numpy
"""
from __future__ import annotations
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import pyigtl


class OpenIGTLinkMessageReceiver:
    """
    Wraps a pyigtl client to receive and dispatch typed OpenIGTLink messages.

    pyigtl's wait_for_message() requires a specific device_name upfront.
    Instead, get_latest_messages() is polled in a loop — it returns a list
    of all messages received since the last call. All returned messages are
    buffered internally; successive read_message() calls drain that buffer
    before polling the network again, so no messages are dropped.

    Usage
    -----
    >>> from igtl_connect import igtl_connect, igtl_disconnect
    >>> from OpenIGTLinkMessageReceiver import OpenIGTLinkMessageReceiver
    >>> client = igtl_connect('127.0.0.1', 18944)
    >>> receiver = OpenIGTLinkMessageReceiver(client, on_string=my_cb)
    >>> receiver.read_message()
    >>> igtl_disconnect(client)
    """

    def __init__(
        self,
        client: pyigtl.OpenIGTLinkClient,
        on_status:    Optional[Callable[[str, str], None]]          = None,
        on_string:    Optional[Callable[[str, str], None]]          = None,
        on_transform: Optional[Callable[[str, np.ndarray], None]]   = None,
        on_point:     Optional[Callable[[str, np.ndarray], None]]   = None,
        on_image:     Optional[Callable[[str, dict], None]]         = None,
        timeout: float = 5.0,
        poll_interval: float = 0.01,
    ) -> None:
        """
        Parameters
        ----------
        client         : live client from igtl_connect()
        on_status      : callback(device_name: str, message: str)
        on_string      : callback(device_name: str, text: str)
        on_transform   : callback(device_name: str, matrix: ndarray [4x4])
        on_point       : callback(device_name: str, point_list: ndarray [Nx3])
        on_image       : callback(device_name: str, image: dict)
                           image dict keys: 'matrix', 'spacing', 'origin',
                           'world_coordinate_system'
        timeout        : total seconds to wait for each incoming message
        poll_interval  : seconds between get_latest_messages() polls
        """
        self._client        = client
        self._timeout       = timeout
        self._poll_interval = poll_interval
        self._pending: deque = deque()
        self._cb = {
            'STATUS':    on_status    or (lambda n, m: None),
            'STRING':    on_string    or (lambda n, m: None),
            'TRANSFORM': on_transform or (lambda n, t: None),
            'POINT':     on_point     or (lambda n, p: None),
            'IMAGE':     on_image     or (lambda n, i: None),
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def read_message(self):
        """
        Poll until one message arrives, dispatch it to the matching callback,
        and return the result.

        ALL messages returned by get_latest_messages() are buffered internally;
        subsequent calls drain that buffer before polling the network again,
        so no messages are silently dropped.

        Returns
        -------
        (device_name: str, data: any) - or (None, None) on timeout.
        """
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            # Refill from the network only when local buffer is empty
            if not self._pending:
                new_msgs = self._client.get_latest_messages()
                self._pending.extend(new_msgs)

            if self._pending:
                msg      = self._pending.popleft()
                msg_type = msg.message_type.strip().upper()
                name     = msg.device_name.strip('\x00').strip()

                handlers = {
                    'STATUS':    self._handle_status,
                    'STRING':    self._handle_string,
                    'TRANSFORM': self._handle_transform,
                    'POINT':     self._handle_point,
                    'IMAGE':     self._handle_image,
                }
                handler = handlers.get(msg_type)
                if handler is None:
                    print(f"Currently unsupported message type: {msg_type}")
                    return name, None
                return handler(msg, name)

            time.sleep(self._poll_interval)

        print("Timeout: no message received.")
        return None, None

    # ------------------------------------------------------------------
    # Per-type handlers
    # ------------------------------------------------------------------

    def _handle_status(self, msg, name: str):
        # Slicer currently sends all-zero STATUS messages; treat body as empty
        text = ''
        self._cb['STATUS'](name, text)
        return name, text

    def _handle_string(self, msg, name: str):
        text = msg.string
        self._cb['STRING'](name, text)
        return name, text

    def _handle_transform(self, msg, name: str):
        matrix = np.array(msg.matrix, dtype=np.float64)
        self._cb['TRANSFORM'](name, matrix)
        return name, matrix

    def _handle_point(self, msg, name: str):
        # pyigtl PointMessage unpacks point data into msg.positions:
        # a list of (x, y, z) tuples (set by _unpack_content).
        point_list = np.array(msg.positions, dtype=np.float64)
        self._cb['POINT'](name, point_list)
        return name, point_list

    def _handle_image(self, msg, name: str):
        print(f"IMAGE message received: {name}")
        # pyigtl ImageMessage unpacks into msg.image and msg.ijk_to_world_matrix.
        # Extract spacing (column norms) and origin (last column) for convenience.
        spacing = [
            float(np.linalg.norm(msg.ijk_to_world_matrix[:3, i]))
            for i in range(3)
        ]
        origin = msg.ijk_to_world_matrix[:3, 3].tolist()
        image = {
            'matrix':                  msg.image,
            'spacing':                 spacing,
            'origin':                  origin,
            'world_coordinate_system': msg.world_coordinate_system,
        }
        self._cb['IMAGE'](name, image)
        return name, image
