"""
igtl_connect.py
Replaces: igtlConnect.m + igtlDisconnect.m

Thin wrappers around pyigtl.OpenIGTLinkClient that provide the same
connect / disconnect interface as the original MATLAB functions.
"""
from __future__ import annotations
import pyigtl


def igtl_connect(hostname: str = 'localhost', port: int = 18944) -> pyigtl.OpenIGTLinkClient:
    """
    Connect to an OpenIGTLink server (e.g. 3D Slicer with OpenIGTLinkIF).

    Parameters
    ----------
    hostname : str   IP address or hostname (default 'localhost')
    port     : int   TCP port              (default 18944)

    Returns
    -------
    pyigtl.OpenIGTLinkClient  -  pass to OpenIGTLinkMessageSender / Receiver
    """
    try:
        client = pyigtl.OpenIGTLinkClient(host=hostname, port=port)
        print(f"Connected to OpenIGTLink server at {hostname}:{port}")
        return client
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to OpenIGTLink server at {hostname}:{port}.\n{exc}"
        ) from exc


def igtl_disconnect(client: pyigtl.OpenIGTLinkClient) -> None:
    """
    Disconnect from the OpenIGTLink server.

    Parameters
    ----------
    client : pyigtl.OpenIGTLinkClient  -  object returned by igtl_connect()
    """
    if client is not None:
        client.disconnect()
        print("Disconnected from OpenIGTLink server.")
