# -*- coding: utf-8 -*-
"""CBAM（Convolutional Block Attention Module，卷积注意力模块）+ Ultralytics 动态注册。

参考文献：Woo S, Park J, Lee J Y, et al. CBAM: Convolutional Block Attention Module. ECCV 2018.

本文件做两件事：
    1. 完整实现 CBAM：通道注意力 ChannelAttention + 空间注意力 SpatialAttention；
    2. 把 CBAM 动态注册进 ultralytics，使模型 YAML 里可以直接写 CBAM 这一层，
       全程不修改 ultralytics 源码（site-packages 下一个字节都不改）。

--------------------------------------------------------------------------------
动态注册的原理（论文/答辩可直接引用这段说明）
--------------------------------------------------------------------------------
ultralytics 解析模型 YAML 的入口是 ultralytics/nn/tasks.py::parse_model()，
它把 YAML 里写的模块名（如 Conv、C2f、SPPF）当作字符串，
在当前模块的全局命名空间里查找对应的类（源码写法即 globals()[m]）。

因此，只要在构建模型之前，把自定义的 CBAM 类塞进 ultralytics.nn.tasks
的全局命名空间，YAML 中的 CBAM 就能被正常解析、实例化、训练、保存，
等同于 ultralytics 原生支持的网络层，且无需改动其任何源码文件。

--------------------------------------------------------------------------------
用法
--------------------------------------------------------------------------------
训练（推荐直接使用同目录下的 train_cbam.py，已经封装好）：

    import cbam                                  # 导入即自动完成注册
    from ultralytics import YOLO
    model = YOLO("code/yolov8n-CBAM.yaml")
    model.load("yolov8n.pt")                     # 迁移 COCO 预训练权重
    model.train(data="datasets/tomato/data.yaml", epochs=100, ...)

--------------------------------------------------------------------------------
与 ultralytics 自带 CBAM 的区别（避免答辩时被问住）
--------------------------------------------------------------------------------
本版本 ultralytics（8.4.x）在 ultralytics.nn.modules.conv 中自带了另一个
CBAM 实现，它参考 RTMDet：通道注意力只用平均池化、没有压缩 MLP；
而且它没有被 ultralytics.nn.tasks 导入，所以无法直接在模型 YAML 中使用。
本文件按 ECCV 2018 原论文实现（平均池化 + 最大池化、共享压缩 MLP），
并且只注册到 ultralytics.nn.tasks，不改动 ultralytics.nn.modules 中的同名类，
避免两者混用导致加载到错误的结构。

推理 / Flask Web 服务（重要）：
    训练得到的 best.pt 里保存的是本文件（cbam.py）中的类，
    因此加载权重前必须先让 cbam 模块可被导入（例如先 import cbam），
    否则会报 ModuleNotFoundError: No module named 'cbam'
    （若环境中存在同名类还会出现“加载成功但结构不对”的隐患，所以务必先注册）。

    >>> from cbam import load_cbam_model
    >>> model = load_cbam_model("runs/detect/cbam/weights/best.pt")
    >>> results = model.predict("test.jpg")

自检（不训练，只验证模块与注册是否正常）：

    python code/cbam.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "register_cbam",
    "is_registered",
    "load_cbam_model",
]


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力子模块（Channel Attention Module）。

    同时使用全局平均池化和全局最大池化压缩空间信息（两者互补：
    平均池化保留整体响应，最大池化保留最显著响应），
    再共享同一个两层 MLP（用 1x1 卷积实现）生成通道权重向量。

    Args:
        channels (int): 输入（输出）特征图的通道数。
        reduction (int): 压缩比 r，瓶颈层通道数为 channels // r。
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)  # 设下限，避免通道数很小时瓶颈层过窄
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 两个分支共享同一个 MLP（与原论文一致）
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状为 (N, C, 1, 1) 的通道注意力权重。"""
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力子模块（Spatial Attention Module）。

    沿通道维度分别取平均和最大值，拼成 2 通道特征图，
    再用一个大卷积核（默认 7x7）卷积 + Sigmoid 生成空间权重图。

    Args:
        kernel_size (int): 卷积核大小，原论文取 3 或 7，推荐 7。
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(f"kernel_size 只支持 3 或 7，当前为 {kernel_size}")
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """按原论文的初始化方式初始化卷积核权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状为 (N, 1, H, W) 的空间注意力权重。"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """CBAM：先做通道注意力，再做空间注意力，逐元素相乘实现特征重标定。

    计算流程（输入特征图 F）：
        F'  = F * Mc(F)      # Mc 为通道注意力权重
        F'' = F' * Ms(F')    # Ms 为空间注意力权重

    输入输出特征图的尺寸与通道数完全一致，因此可以像“插件”一样
    插在网络的任意位置，不改变原有网络结构的形状。

    Args:
        c1 (int): 输入通道数（与输出通道数相同）。
        reduction (int): 通道注意力的压缩比，默认 16。
        kernel_size (int): 空间注意力的卷积核大小，默认 7。

    Examples:
        >>> cbam = CBAM(256)
        >>> y = cbam(torch.randn(1, 256, 20, 20))
        >>> y.shape
        torch.Size([1, 256, 20, 20])
    """

    def __init__(self, c1: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.c1 = c1
        self.channel_attention = ChannelAttention(c1, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


# ---------------------------------------------------------------------------
# 动态注册到 ultralytics（不修改源码）
# ---------------------------------------------------------------------------

# 需要注册的类名 -> 类对象
_CBAM_REGISTRY = {
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention,
    "CBAM": CBAM,
}

# 注册的目标命名空间：
#   ultralytics.nn.tasks —— parse_model() 解析 YAML 时查找网络层类的地方（必须且足够）
# 说明：ultralytics.nn.modules 中已有此版本自带的同名类，刻意不改动它，避免混淆。
_TARGET_NAMESPACES = ("ultralytics.nn.tasks",)


def register_cbam(verbose: bool = True) -> bool:
    """把 CBAM 及其子模块动态注册进 ultralytics 的命名空间（幂等，可重复调用）。

    Args:
        verbose (bool): 是否打印注册信息。

    Returns:
        (bool): 注册成功返回 True。
    """
    import importlib

    for module_name in _TARGET_NAMESPACES:
        module = importlib.import_module(module_name)
        for cls_name, cls in _CBAM_REGISTRY.items():
            setattr(module, cls_name, cls)

    if verbose:
        print(f"[CBAM] 动态注册完成 -> {'、'.join(_TARGET_NAMESPACES)}（未修改 ultralytics 源码）")
    return True


def is_registered() -> bool:
    """检查 CBAM 是否已经注册到 ultralytics。"""
    try:
        import ultralytics.nn.tasks as tasks
    except ImportError:  # 未安装 ultralytics
        return False
    return getattr(tasks, "CBAM", None) is CBAM


def load_cbam_model(weights: str | Path, verbose: bool = False):
    """加载含 CBAM 的权重文件（推理 / Web 服务用，自动完成注册）。

    Args:
        weights (str | Path): best.pt / last.pt 等权重路径。
        verbose (bool): 是否打印注册信息。

    Returns:
        (ultralytics.YOLO): 可直接调用 .predict() / .val() 的模型对象。
    """
    register_cbam(verbose=verbose)
    from ultralytics import YOLO

    return YOLO(str(weights))


# 导入本模块即自动完成注册，避免使用者忘记调用
register_cbam(verbose=False)


# ---------------------------------------------------------------------------
# 自检：python code/cbam.py
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """构建基线模型与 CBAM 模型，对比结构、参数量、计算量，并验证前向传播。"""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import model_info

    cbam_yaml = Path(__file__).with_name("yolov8n-CBAM.yaml")
    if not cbam_yaml.exists():
        raise FileNotFoundError(f"未找到模型配置文件：{cbam_yaml}")

    print("\n" + "=" * 72)
    print("CBAM 模块自检")
    print("=" * 72)
    print(f"是否已注册到 ultralytics：{is_registered()}")

    base_model = DetectionModel("yolov8n.yaml", nc=10, verbose=False).eval()
    cbam_model = DetectionModel(str(cbam_yaml), nc=10, verbose=False).eval()

    n_cbam = sum(1 for m in cbam_model.modules() if isinstance(m, CBAM))
    print(f"CBAM 层数量：{n_cbam}")

    # 前向传播检查
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 320, 320)
        out = cbam_model(dummy)
        cbam_layer = next(m for m in cbam_model.modules() if isinstance(m, CBAM))
        feat = torch.randn(2, cbam_layer.c1, 20, 20)
        ch_w = cbam_layer.channel_attention(feat)
        sp_w = cbam_layer.spatial_attention(feat)
    n_out = len(out) if isinstance(out, (list, tuple)) else 1
    print(f"前向传播输出类型：{type(out).__name__}，输出数量：{n_out}")
    print(f"通道注意力权重形状 {tuple(ch_w.shape)}，取值范围 ({ch_w.min():.3f}, {ch_w.max():.3f})")
    print(f"空间注意力权重形状 {tuple(sp_w.shape)}，取值范围 ({sp_w.min():.3f}, {sp_w.max():.3f})")

    n_l_b, n_p_b, _, flops_b = model_info(base_model, verbose=True)
    n_l_c, n_p_c, _, flops_c = model_info(cbam_model, verbose=True)
    print("\n" + "-" * 72)
    print(f"{'模型':<16}{'层数':>8}{'参数量':>12}{'GFLOPs':>10}")
    print(f"{'YOLOv8n（基线）':<16}{n_l_b:>8}{n_p_b:>12,}{flops_b:>10.1f}")
    print(f"{'YOLOv8n-CBAM':<16}{n_l_c:>8}{n_p_c:>12,}{flops_c:>10.1f}")
    print(f"{'增量':<16}{n_l_c - n_l_b:>8}{n_p_c - n_p_b:>+12,}{flops_c - flops_b:>+10.1f}")
    print("-" * 72)
    print("自检通过：CBAM 已成功接入 ultralytics，可以开始训练。")


if __name__ == "__main__":
    _self_test()
