"""Wrapper for segmentation_models_pytorch (smp) models.

Provides a unified interface for external comparison architectures:
    - Unet (ResNet34 encoder)
    - DeepLabV3Plus (MobileNetV2 encoder)
    - FPN (EfficientNet-B0 encoder)

These models natively support arbitrary in_channels via smp's API.

Requires: pip install segmentation-models-pytorch
"""

import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None


class SMPModelWrapper(nn.Module):
    """Wrapper for segmentation_models_pytorch models.

    Adapts smp models to match the project's model interface.

    Args:
        arch: SMP architecture name ("Unet", "DeepLabV3Plus", "FPN", "PSPNet").
        encoder_name: SMP encoder name ("resnet34", "efficientnet-b0", "mobilenet_v2").
        in_channels: Number of input channels (default 9 for MSI).
        num_classes: Number of segmentation classes.
        encoder_weights: Pretrained weights ("imagenet" or None).
    """

    def __init__(self, arch="Unet", encoder_name="resnet34",
                 in_channels=9, num_classes=2, encoder_weights="imagenet",
                 first_layer_pretrained=True):
        super().__init__()

        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required for SMP models. "
                "Install it with: pip install segmentation-models-pytorch"
            )

        # Get the model class from smp
        model_cls = getattr(smp, arch, None)
        if model_cls is None:
            raise ValueError(
                f"Unknown SMP architecture '{arch}'. "
                f"Available: Unet, DeepLabV3Plus, FPN, PSPNet, etc."
            )

        self.model = model_cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        # Optionally randomize only the encoder's input conv while keeping the
        # deeper backbone pretrained (removes first-conv pretraining as a confound).
        if encoder_weights is not None and not first_layer_pretrained:
            for m in self.model.encoder.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    break

    def forward(self, x):
        return self.model(x)


def build_smp_model(cfg):
    """Build SMP model from config."""
    model_cfg = cfg["model"]
    return SMPModelWrapper(
        arch=model_cfg.get("smp_arch", "Unet"),
        encoder_name=model_cfg.get("smp_encoder", "resnet34"),
        in_channels=cfg["data"].get("num_channels", 9),
        num_classes=model_cfg.get("num_classes", 2),
        encoder_weights="imagenet" if model_cfg.get("encoder_pretrained", True) else None,
        first_layer_pretrained=model_cfg.get("first_layer_pretrained", True),
    )
