"""
OpenIGTLinkMessageSender.py
Replaces: OpenIGTLinkMessageSender.m

Sends STRING, TRANSFORM, POINT, and IMAGE messages over OpenIGTLink
using Protocol v3 with the same Slicer-compatible metadata keys
(MRMLNodeName, Status) as the original MATLAB implementation.

Requires: pyigtl >= 0.2.0, numpy
"""
from __future__ import annotations
from typing import Union

import numpy as np
import pyigtl


class OpenIGTLinkMessageSender:
    """
    Sends typed OpenIGTLink messages to 3D Slicer via a connected pyigtl client.

    All messages include Protocol-v3 metadata with the same MRMLNodeName /
    Status keys expected by Slicer's OpenIGTLinkIF.

    Usage
    -----
    >>> from igtl_connect import igtl_connect, igtl_disconnect
    >>> from OpenIGTLinkMessageSender import OpenIGTLinkMessageSender
    >>> client = igtl_connect('127.0.0.1', 18944)
    >>> sender = OpenIGTLinkMessageSender(client)
    >>> sender.send_string('MyDevice', 'Hello Slicer!')
    >>> igtl_disconnect(client)
    """

    def __init__(self, client: pyigtl.OpenIGTLinkClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # STRING
    # ------------------------------------------------------------------

    def send_string(self, device_name: str, text: str) -> bool:
        """
        Send a STRING message.

        Slicer metadata:  MRMLNodeName='Text', Status='OK'
        """
        msg = pyigtl.StringMessage(text, device_name=device_name)
        msg.metadata = {'MRMLNodeName': 'Text', 'Status': 'OK'}
        return self._send(msg)

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------

    def send_transform(self, device_name: str, matrix: np.ndarray) -> bool:
        """
        Send a TRANSFORM message.

        Parameters
        ----------
        matrix : (4, 4) array_like, float32
            Homogeneous transformation matrix.

        Slicer metadata:  MRMLNodeName='LinearTransform'
        """
        mat = np.asarray(matrix, dtype=np.float32)
        if mat.shape != (4, 4):
            raise ValueError("Transform matrix must be shape (4, 4).")
        msg = pyigtl.TransformMessage(mat, device_name=device_name)
        msg.metadata = {'MRMLNodeName': 'LinearTransform'}
        return self._send(msg)

    # ------------------------------------------------------------------
    # POINT
    # ------------------------------------------------------------------

    def send_point(self, device_name: str, point_list: np.ndarray) -> bool:
        """
        Send a POINT message (fiducial list).

        Parameters
        ----------
        point_list : (N, 3) array_like, float32
            XYZ coordinates in mm.

        Slicer metadata:  MRMLNodeName='MarkupsFiducial', Status='OK'

        Note: each point is packed as 136 bytes per the OpenIGTLink POINT spec:
              name(64) + group(32) + RGBA(4) + XYZ(12) + diameter(4) + owner(20)
        """
        pts = np.asarray(point_list, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("point_list must be shape (N, 3).")

        points = []
        for i, xyz in enumerate(pts):
            points.append({
                'name':       f"{device_name}-{i + 1}",
                'group_name': 'Selected',
                'rgba':       [255, 127, 127, 255],
                'position':   xyz.tolist(),
                'radius':     0.0,
                'owner':      '',
            })

        msg = pyigtl.PointMessage(points=points, device_name=device_name)
        msg.metadata = {'MRMLNodeName': 'MarkupsFiducial', 'Status': 'OK'}
        return self._send(msg)

    # ------------------------------------------------------------------
    # IMAGE
    # ------------------------------------------------------------------

    def send_image(
        self,
        device_name: str,
        image_input: Union[np.ndarray, dict],
    ) -> bool:
        """
        Send an IMAGE message.

        Parameters
        ----------
        image_input : ndarray or dict
            ndarray  - 2-D, 3-D, or 4-D (last axis = channels).
            dict with keys:
                'matrix'      - numpy array  (required)
                'origin'      - [Px, Py, Pz] in mm        (default [0,0,0])
                'orientation' - (3,3) float32 array
                                rows = T, S, N direction vectors;
                                vector norms define voxel spacing in mm
                                (default identity -> 1 mm isotropic)
                'coordinate'  - 1=RAS, 2=LPS  (informational, default 2)

        Slicer metadata:  MRMLNodeName='ScalarVolume'
        """
        if isinstance(image_input, np.ndarray):
            mat    = image_input
            origin = [0.0, 0.0, 0.0]
            orient = np.eye(3, dtype=np.float32)
        elif isinstance(image_input, dict):
            mat    = np.asarray(image_input['matrix'])
            origin = list(image_input.get('origin', [0.0, 0.0, 0.0]))
            orient = np.asarray(
                image_input.get('orientation', np.eye(3)), dtype=np.float32
            )
        else:
            raise TypeError("image_input must be a numpy array or a dict.")

        if mat.ndim < 2 or mat.ndim > 4:
            raise ValueError("Image matrix must be 2-D, 3-D, or 4-D.")

        spacing = [float(np.linalg.norm(orient[r])) for r in range(3)]
        spacing = [s if s > 0 else 1.0 for s in spacing]

        msg = pyigtl.ImageMessage(
            image=mat,
            spacing=spacing,
            origin=origin,
            device_name=device_name,
        )
        msg.metadata = {'MRMLNodeName': 'ScalarVolume'}
        return self._send(msg)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _send(self, msg) -> bool:
        try:
            self._client.send_message(msg)
            return True
        except Exception as exc:
            print(f"Sending OpenIGTLink message failed: {exc}")
            return False
