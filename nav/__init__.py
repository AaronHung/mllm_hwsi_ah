"""nav — 我們自己的 navigation/CL 研究套件。

乾淨原則：
- 不 import MLLM-HWSI 的 model / projector / QPMIL 任何程式。
- 只讀取離線特徵檔（.pt / .h5 / .csv）。
- device 自動選擇（cuda > mps > cpu），同一份 code 在 Mac M1 與 RunPod 跑，不分叉。
"""

from .device import (DeviceInfo, add_device_argument, device_info, get_device,
                     resolve_device, setup_mps)

__all__ = [
    "DeviceInfo",
    "add_device_argument",
    "device_info",
    "get_device",
    "resolve_device",
    "setup_mps",
]
