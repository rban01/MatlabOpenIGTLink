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

        pyigtl PointMessage constructor takes separate keyword arrays:
            positions, names, groups, rgba_colors, diameters, owners

        Slicer metadata:  MRMLNodeName='MarkupsFiducial', Status='OK'
        """
        pts = np.asarray(point_list, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("point_list must be shape (N, 3).")

        n = len(pts)
        names      = [f"{device_name}-{i + 1}" for i in range(n)]
        groups     = ['Selected'] * n
        rgba_colors = [[255, 127, 127, 255]] * n

        msg = pyigtl.PointMessage(
            positions=pts.tolist(),
            names=names,
            groups=groups,
            rgba_colors=rgba_colors,
            device_name=device_name,
        )
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
            ndarray  - 2-D, 3-D, or 4-D (last axis = channels); 1 mm isotropic LPS assumed.
            dict with keys:
                'matrix'      - numpy array  (required)
                'origin'      - [Px, Py, Pz] in mm        (default [0,0,0])
                'orientation' - (3,3) float32 array
                                columns = i, j, k axis directions scaled by voxel spacing
                                (default identity -> 1 mm isotropic)
                'coordinate'  - 1=RAS, 2=LPS  (default 2)

        pyigtl ImageMessage constructor takes:
            image, ijk_to_world_matrix (4x4), world_coordinate_system ('lps'/'ras')

        Slicer metadata:  MRMLNodeName='ScalarVolume'
        """
        if isinstance(image_input, np.ndarray):
            mat                    = image_input
            ijk_to_world           = np.eye(4, dtype=np.float32)
            world_coordinate_system = 'lps'
        elif isinstance(image_input, dict):
            mat    = np.asarray(image_input['matrix'])
            origin = list(image_input.get('origin', [0.0, 0.0, 0.0]))
            orient = np.asarray(
                image_input.get('orientation', np.eye(3)), dtype=np.float32
            )
            coordinate = image_input.get('coordinate', 2)
            world_coordinate_system = 'ras' if coordinate == 1 else 'lps'

            # Build 4x4 ijk_to_world: columns are axis directions (scaled by spacing)
            ijk_to_world = np.eye(4, dtype=np.float32)
            ijk_to_world[:3, :3] = orient
            ijk_to_world[:3,  3] = origin
        else:
            raise TypeError("image_input must be a numpy array or a dict.")

        if mat.ndim < 2 or mat.ndim > 4:
            raise ValueError("Image matrix must be 2-D, 3-D, or 4-D.")

        msg = pyigtl.ImageMessage(
            image=mat,
            ijk_to_world_matrix=ijk_to_world,
            world_coordinate_system=world_coordinate_system,
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
