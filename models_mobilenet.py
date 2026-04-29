import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

'''
1. Mel Spectrogram CNN (using MobileNetV3-Small)
'''
class MelCNN(nn.Module):
    def __init__(self, num_classes=3, n_mels=128, time_frames=128, **kwargs):
        super(MelCNN, self).__init__()
        
        # Load unpretrained MobileNetV3-Small (for fair comparison)
        self.backbone = models.mobilenet_v3_small(weights=None)
        
        # 1. Modify the first conv layer to accept 1-channel input (original is 3-channel RGB)
        original_first_layer = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )
        
        # 2. Modify the final classifier to output num_classes (3 classes)
        original_classifier = self.backbone.classifier[3]
        self.backbone.classifier[3] = nn.Linear(
            in_features=original_classifier.in_features, 
            out_features=num_classes
        )
        
    def forward(self, x):
        # x shape: (Batch, 1, 128, 128)
        return self.backbone(x)

# Alias mapping remains unchanged
MelLDSCNN = MelCNN


'''
2. Mel Spectrogram and Hilbert Image CNN (using MobileNetV3-Small)
'''

# ==========================================
# MobileNetV3 Core Component Definitions
# ==========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention module (MobileNetV3 style)"""
    def __init__(self, exp_channels):
        super(SEBlock, self).__init__()
        # In MobileNetV3, SE module usually compresses channels to 1/4 of original
        squeeze_channels = _make_divisible(exp_channels // 4, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(exp_channels, squeeze_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, exp_channels, 1, bias=True),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        return x * self.fc(x)

def _make_divisible(v, divisor, min_value=None):
    """Ensure channel count is divisible by 8 (hardware acceleration friendly)"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class InvertedResidual(nn.Module):
    """Core of MobileNetV3: Inverted Residual Block"""
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        act_layer = nn.Hardswish if use_hs else nn.ReLU
        
        layers =[]
        # 1. Expand (increase dimension)
        if exp_channels != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, exp_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(exp_channels),
                act_layer(inplace=True)
            ])
            
        # 2. Depthwise (depthwise separable convolution)
        layers.extend([
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride=stride, 
                      padding=(kernel_size - 1) // 2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            act_layer(inplace=True)
        ])
        
        # 3. Squeeze-and-Excitation (attention mechanism)
        if use_se:
            layers.append(SEBlock(exp_channels))
            
        # 4. Project (dimensionality reduction projection, note no activation function here)
        layers.extend([
            nn.Conv2d(exp_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

# ==========================================
# MobileNetV3 architecture designed specifically for MelHilbert (strictly aligned to 1.52M parameters, solving information bottleneck)
# ==========================================

class MelHilbertMobileNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=128, **kwargs):
        super(MelHilbertMobileNet, self).__init__()
        
        # 1. Initial layer: alleviate information bottleneck
        # Input: (128, 8, 16) -> Output: (64, 8, 16)
        # No longer extremely compressed to 16, but retain 64 channels to preserve sufficient frequency features
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.Hardswish(inplace=True)
        )
        
        # 2. Build InvertedResidual block sequence (customized channel width, controlling total parameter count)
        # Parameter order: in_c, exp_c, out_c, kernel, stride, use_se, use_hs
        self.bnecks = nn.Sequential(
            # Stage 1: spatial size unchanged (8x16)
            InvertedResidual(96, 128, 96, 3, 1, True, True),
            InvertedResidual(96, 128, 96, 3, 1, True, True),
            
            # Stage 2: 1st downsampling (4x8)
            InvertedResidual(96, 192, 96, 3, 2, True, True),
            InvertedResidual(96, 288, 128, 3, 1, True, True),
            
            # Stage 3: 2nd downsampling (2x4)
            InvertedResidual(128, 384, 192, 3, 2, True, True),
            InvertedResidual(192, 576, 256, 3, 1, True, True),
        )
        
        # 3. Final feature extraction and classifier
        self.conv_last = nn.Sequential(
            nn.Conv2d(256, 576, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1) # 2x4 -> 1x1
        
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x):
        # x shape: (Batch, 128, 8, 16)
        x = self.first_conv(x)
        x = self.bnecks(x)
        x = self.conv_last(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

# Alias mapping
MelHilbertLDSCNN = MelHilbertMobileNet
MelHilbertCNN = MelHilbertMobileNet


'''
3. Mel Spectrogram and Hilbert Time-axis CNN (using MobileNetV3-Small)
'''
class MelHilbertTimeCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=128, **kwargs):
        super(MelHilbertTimeCNN, self).__init__()
        
        # 1. Initial layer: alleviate information bottleneck
        # Input: (128, 8, 16) -> Output: (64, 8, 16)
        # No longer extremely compressed to 16, but retain 64 channels to preserve sufficient frequency features
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.Hardswish(inplace=True)
        )
        
        # 2. Build InvertedResidual block sequence (customized channel width, controlling total parameter count)
        # Parameter order: in_c, exp_c, out_c, kernel, stride, use_se, use_hs
        self.bnecks = nn.Sequential(
            # Stage 1: spatial size unchanged (8x16)
            InvertedResidual(96, 128, 96, 3, 1, True, True),
            InvertedResidual(96, 128, 96, 3, 1, True, True),
            
            # Stage 2: 1st downsampling (4x8)
            InvertedResidual(96, 192, 96, 3, 2, True, True),
            InvertedResidual(96, 288, 128, 3, 1, True, True),
            
            # Stage 3: 2nd downsampling (2x4)
            InvertedResidual(128, 384, 192, 3, 2, True, True),
            InvertedResidual(192, 576, 256, 3, 1, True, True),
        )
        
        # 3. Final feature extraction and classifier
        self.conv_last = nn.Sequential(
            nn.Conv2d(256, 576, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1) # 2x4 -> 1x1
        
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x):
        # x shape: (Batch, 128, 8, 16)
        x = self.first_conv(x)
        x = self.bnecks(x)
        x = self.conv_last(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

# Alias mapping remains unchanged
MelHilbertTimeLDSCNN = MelHilbertTimeCNN

'''
4. Single-channel Hilbert Image CNN (using MobileNetV3-Small)
'''
class SignalHilbertCNN(nn.Module):
    def __init__(self, num_classes=3, hilbert_height=128, hilbert_width=128, **kwargs):
        super(SignalHilbertCNN, self).__init__()
        
        # Load unpretrained MobileNetV3-Small
        self.backbone = models.mobilenet_v3_small(weights=None)
        
        # 1. Modify the first conv layer to accept 1-channel input
        original_first_layer = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )
        
        # 2. Modify the final classifier
        original_classifier = self.backbone.classifier[3]
        self.backbone.classifier[3] = nn.Linear(
            in_features=original_classifier.in_features, 
            out_features=num_classes
        )
        
    def forward(self, x):
        # x shape: (Batch, 1, 128, 128)
        return self.backbone(x)

# Alias mapping remains unchanged
SignalHilbertLDSCNN = SignalHilbertCNN


def convert_2d_to_1d(module):
    """
    Recursively replace 2D layers in the model with equivalent 1D layers.
    This ensures the 1D model has exactly the same macroscopic topology as the 2D model,
    providing a strict control variable for factorial analysis.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Extract 2D conv parameters, using the first dimension as 1D parameter
            new_conv = nn.Conv1d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size[0],
                stride=child.stride[0],
                padding=child.padding[0],
                groups=child.groups,
                bias=(child.bias is not None)
            )
            setattr(module, name, new_conv)
        elif isinstance(child, nn.BatchNorm2d):
            new_bn = nn.BatchNorm1d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats
            )
            setattr(module, name, new_bn)
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            # MobileNetV3 SE module and final pooling layer use this
            new_pool = nn.AdaptiveAvgPool1d(1)
            setattr(module, name, new_pool)
        else:
            # Recursively process submodules (e.g. Sequential, InvertedResidual, etc.)
            convert_2d_to_1d(child)
    return module

'''
5. 1D Signal CNN (using 1D version of MobileNetV3-Small)
'''
class SignalCNN(nn.Module):
    def __init__(self, num_classes=3, signal_length=1024, **kwargs):
        super(SignalCNN, self).__init__()
        
        # 1. Load standard MobileNetV3-Small (no pretraining)
        self.backbone = models.mobilenet_v3_small(weights=None)
        
        # 2. Core trick: convert the entire network from 2D to 1D
        self.backbone = convert_2d_to_1d(self.backbone)
        
        # 3. Modify the first conv layer to accept 1-channel input (original is 3-channel)
        original_first_layer = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv1d(
            in_channels=1, 
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size[0],
            stride=original_first_layer.stride[0],
            padding=original_first_layer.padding[0],
            bias=(original_first_layer.bias is not None)
        )
        
        # 4. Modify the final classifier to output num_classes
        original_classifier = self.backbone.classifier[3]
        self.backbone.classifier[3] = nn.Linear(
            in_features=original_classifier.in_features, 
            out_features=num_classes
        )
        
    def forward(self, x):
        # x shape: (Batch, 1, 1024)
        return self.backbone(x)

# Alias mapping remains unchanged
SignalLDSCNN = SignalCNN