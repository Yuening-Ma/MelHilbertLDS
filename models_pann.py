import torch.nn as nn
import torch.nn.functional as F

'''
SignalHilbertCNN, SignalHilbertLDSCNN,
MelCNN, MelLDSCNN, MelLDS1CNN, 
MelHilbertCNN, MelHilbertLDSCNN, MelHilbertLDS1CNN, 
MelHilbertTimeCNN, MelHilbertTimeLDSCNN, MelHilbertTimeLDS1CNN
'''

def make_panns_block(in_channels, out_channels, pool_size=2):
    """
    Build a PANNs-style convolution block using only torch.nn operators
    """
    layers = [
        # PANNs CNN6 uses 5x5 conv, padding=2 to ensure size unchanged after convolution
        nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool_size > 1:
        layers.append(nn.AvgPool2d(kernel_size=pool_size))
    layers.append(nn.Dropout(0.5))
    
    return nn.Sequential(*layers)

'''
1. Mel Spectrogram CNN
'''
class MelCNN(nn.Module):
    def __init__(self, num_classes=3, **kwargs):
        super(MelCNN, self).__init__()
        
        # 4 standard PANNs convolution blocks
        self.block1 = make_panns_block(1, 64, pool_size=2)   # -> 64 x 64 x 64
        self.block2 = make_panns_block(64, 128, pool_size=2) # -> 128 x 32 x 32
        self.block3 = make_panns_block(128, 256, pool_size=2)# -> 256 x 16 x 16
        self.block4 = make_panns_block(256, 512, pool_size=2)# -> 512 x 8 x 8
    
        self.avg_global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_global_pool(x).flatten(1) + self.max_global_pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)

MelLDSCNN = MelCNN


'''
2. Mel Spectrogram and Hilbert Image CNN
'''
class MelHilbertCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=128, **kwargs):
        super(MelHilbertCNN, self).__init__()
        
        # First block: no pooling, preserve 8x16 spatial resolution, while processing 128 frequency channels
        self.block1 = make_panns_block(in_channels, 64, pool_size=1) # -> 64 x 8 x 16
        
        # Last three blocks: normal pooling
        self.block2 = make_panns_block(64, 128, pool_size=2)         # -> 128 x 4 x 8
        self.block3 = make_panns_block(128, 256, pool_size=2)        # -> 256 x 2 x 4
        self.block4 = make_panns_block(256, 512, pool_size=2)        # -> 512 x 1 x 2
        
        self.avg_global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_global_pool(x).flatten(1) + self.max_global_pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)

MelHilbertLDSCNN = MelHilbertCNN


'''
3. Mel Spectrogram and Hilbert Image and Time-axis CNN
'''
class MelHilbertTimeCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=128, **kwargs):
        super(MelHilbertTimeCNN, self).__init__()
        
        # First block: no pooling, preserve 8x16 spatial resolution, while processing 128 frequency channels
        self.block1 = make_panns_block(in_channels, 64, pool_size=1) # -> 64 x 8 x 16
        
        # Last three blocks: normal pooling
        self.block2 = make_panns_block(64, 128, pool_size=2)         # -> 128 x 4 x 8
        self.block3 = make_panns_block(128, 256, pool_size=2)        # -> 256 x 2 x 4
        self.block4 = make_panns_block(256, 512, pool_size=2)        # -> 512 x 1 x 2
        
        self.avg_global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_global_pool(x).flatten(1) + self.max_global_pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        return self.fc_out(x)

MelHilbertTimeLDSCNN = MelHilbertTimeCNN

'''
4. Single-channel Hilbert Image CNN
'''
class SignalHilbertCNN(nn.Module):
    def __init__(self, num_classes=3, hilbert_height=128, hilbert_width=128, **kwargs):
        super(SignalHilbertCNN, self).__init__()
        
        # 4 standard PANNs convolution blocks
        self.block1 = make_panns_block(1, 64, pool_size=2)   # -> 64 x 64 x 64
        self.block2 = make_panns_block(64, 128, pool_size=2) # -> 128 x 32 x 32
        self.block3 = make_panns_block(128, 256, pool_size=2)# -> 256 x 16 x 16
        self.block4 = make_panns_block(256, 512, pool_size=2)# -> 512 x 8 x 8
        
        self.avg_global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_global_pool(x).flatten(1) + self.max_global_pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)

SignalHilbertLDSCNN = SignalHilbertCNN

'''
5. 1D Signal CNN
'''
def make_panns_1d_block(in_channels, out_channels, pool_size=2):
    """
    Build a 1D PANNs-style convolution block: standard conv -> BN -> ReLU -> MaxPool
    """
    layers = [
        # Use kernel_size=5, padding=2 to keep length unchanged
        nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool_size > 1:
        layers.append(nn.AvgPool1d(kernel_size=pool_size))
    layers.append(nn.Dropout(0.5))
    
    return nn.Sequential(*layers)


class SignalCNN(nn.Module):
    def __init__(self, num_classes=3, signal_length=1024, **kwargs):
        super(SignalCNN, self).__init__()
        
        # Gradually increase channels and depth to align parameter count
        self.block1 = make_panns_1d_block(1, 64, pool_size=2)    # -> 64 x 512
        self.block2 = make_panns_1d_block(64, 128, pool_size=2)  # -> 128 x 256
        self.block3 = make_panns_1d_block(128, 128, pool_size=2) # -> 256 x 128
        self.block4 = make_panns_1d_block(128, 256, pool_size=2) # -> 512 x 64
        self.block5 = make_panns_1d_block(256, 256, pool_size=2) # -> 512 x 32
        self.block6 = make_panns_1d_block(256, 512, pool_size=2)# -> 1024 x 16
        self.block7 = make_panns_1d_block(512, 512, pool_size=2)# -> 1024 x 8
        
        # Global pooling retains information from 8 time points
        self.avg_global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.fc1 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (Batch, 1, 1024)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        
        x = self.avg_global_pool(x).flatten(1) + self.max_global_pool(x).flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)

        
SignalLDSCNN = SignalCNN

# if __name__ == "__main__":
    
#     import torch
#     from torchsummary import summary

#     model = SignalCNN().to("cuda")
#     summary(model, (1, 1024))