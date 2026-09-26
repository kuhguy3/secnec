"""Direction of arrival from a ReSpeaker USB Mic Array v2.0 (XMOS XVF-3000).

Reads the firmware's DOAANGLE parameter over USB vendor control transfers,
the same way respeaker/usb_4_mic_array's tuning.py does. Requires pyusb and a
udev rule granting access to 2886:0018.
"""
from __future__ import annotations

import logging
import struct

log = logging.getLogger(__name__)

VID, PID = 0x2886, 0x0018
DOAANGLE = (21, 0)  # (parameter id, offset), int, read-only, 0..359


class RespeakerDoa:
    def __init__(self):
        import usb.core
        import usb.util

        self.usb = usb
        self.dev = usb.core.find(idVendor=VID, idProduct=PID)
        if self.dev is None:
            raise RuntimeError("ReSpeaker USB Mic Array v2.0 not found")

    def angle(self) -> int | None:
        pid, offset = DOAANGLE
        cmd = 0x80 | offset | 0x40  # read, int
        try:
            resp = self.dev.ctrl_transfer(
                self.usb.util.CTRL_IN | self.usb.util.CTRL_TYPE_VENDOR | self.usb.util.CTRL_RECIPIENT_DEVICE,
                0, cmd, pid, 8, 1000)
        except Exception:  # noqa: BLE001 - USB hiccups must never stop detection
            log.debug("DOA read failed", exc_info=True)
            return None
        return struct.unpack("ii", bytes(resp))[0]


def in_sectors(angle: int | None, sectors: list[list[int]]) -> bool:
    """True if no sectors are configured, the angle is unknown, or it lies in any [lo, hi] sector.

    Sectors may wrap around 0, e.g. [330, 30].
    """
    if not sectors or angle is None:
        return True
    for lo, hi in sectors:
        if (lo <= angle <= hi) if lo <= hi else (angle >= lo or angle <= hi):
            return True
    return False
