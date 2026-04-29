import torch.nn as nn
import torch.nn.functional as F

'''
SignalHilbertCNN, SignalHilbertLDSCNN,
MelCNN, MelLDSCNN, MelLDS1CNN, 
MelHilbertCNN, MelHilbertLDSCNN, MelHilbertLDS1CNN, 
MelHilbertTimeCNN, MelHilbertTimeLDSCNN, MelHilbertTimeLDS1CNN
'''

'''
1. Mel频谱CNN
'''
class MelCNN(nn.Module):
    def __init__(self, num_classes=3, n_mels=128, time_frames=128, **kwargs):
        super(MelCNN, self).__init__()
        
        # 第一个深度可分离卷积-BN组合
        self.dwconv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, groups=1)
        self.pwconv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        
        # 第二个深度可分离卷积-BN组合
        self.dwconv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8)
        self.pwconv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 第三个深度可分离卷积-BN组合
        self.dwconv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, groups=16)
        self.pwconv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        
        # 第四个深度可分离卷积-BN组合
        self.dwconv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32)
        self.pwconv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 第五个深度可分离卷积-BN组合
        self.dwconv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pwconv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(128)
        
        # 第六个深度可分离卷积-BN组合
        self.dwconv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.pwconv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(256)
        
        # 第七个深度可分离卷积-BN组合
        self.dwconv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256)
        self.pwconv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)
        self.bn7 = nn.BatchNorm2d(512)
        
        # 计算全连接层的输入维度
        # 经过7次下采样 (stride=2)，尺寸变为原来的 1/128
        h_out = n_mels // 128
        w_out = time_frames // 128
        fc_input_dim = 512 * h_out * w_out
        
        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 深度可分离卷积-BN层
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.dwconv2(x)
        x = self.pwconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.dwconv3(x)
        x = self.pwconv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.dwconv4(x)
        x = self.pwconv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.dwconv5(x)
        x = self.pwconv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        x = self.dwconv6(x)
        x = self.pwconv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        
        x = self.dwconv7(x)
        x = self.pwconv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        # 展平操作
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

MelLDSCNN = MelCNN
'''
2. Mel频谱和Hilbert图像CNN
'''
class MelHilbertCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=128, **kwargs):
        super(MelHilbertCNN, self).__init__()
        
        # # 第一组1×1卷积：减少通道数
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 第一组深度可分离卷积：增大通道数、减小特征图尺寸
        self.dwconv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.pwconv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.bn_dw1 = nn.BatchNorm2d(128)
        
        # 第二组深度可分离卷积：增大通道数、减小特征图尺寸
        self.dwconv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.pwconv2 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=1, stride=2, padding=0)
        self.bn_dw2 = nn.BatchNorm2d(192)
        
        # 第三组深度可分离卷积：增大通道数、减小特征图尺寸
        self.dwconv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192)
        self.pwconv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=1, stride=2, padding=0)
        self.bn_dw3 = nn.BatchNorm2d(384)
        
        # 第四组深度可分离卷积：增大通道数、减小特征图尺寸
        self.dwconv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=384)
        self.pwconv4 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=1, stride=2, padding=0)
        self.bn_dw4 = nn.BatchNorm2d(768)
        
        # 全连接层
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(768*1*1, 96)
        self.bn_fc1 = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, num_classes)
        
    def forward(self, x):

        # # 第一组1×1卷积
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = F.relu(x)

        # 第一组深度可分离卷积
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self.bn_dw1(x)
        x = F.relu(x)
        
        # 第二组深度可分离卷积
        x = self.dwconv2(x)
        x = self.pwconv2(x)
        x = self.bn_dw2(x)
        x = F.relu(x)
        
        # 第三组深度可分离卷积
        x = self.dwconv3(x)
        x = self.pwconv3(x)
        x = self.bn_dw3(x)
        x = F.relu(x)
        
        # 第四组深度可分离卷积
        x = self.dwconv4(x)
        x = self.pwconv4(x)
        x = self.bn_dw4(x)
        x = F.relu(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x) 

        x = self.dropout(x)
        x = self.fc2(x)

        return x

MelHilbertLDSCNN = MelHilbertCNN
'''
3. Mel频谱和Hilbert图像和时间轴CNN
'''
class MelHilbertTimeCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=128, **kwargs):
        super(MelHilbertTimeCNN, self).__init__()
        
        # # 第一组1×1卷积：减少通道数
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 第一组深度可分离卷积：增大通道数、减小特征图尺寸
        self.dwconv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.pwconv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.bn_dw1 = nn.BatchNorm2d(128)
        
        # 第二组深度可分离卷积：增大通道数、减小特征图尺寸
        self.dwconv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.pwconv2 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=1, stride=2, padding=0)
        self.bn_dw2 = nn.BatchNorm2d(192)
        
        # 第三组深度可分离卷积：增大通道数、减小特征图尺寸
        self.dwconv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192)
        self.pwconv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=1, stride=2, padding=0)
        self.bn_dw3 = nn.BatchNorm2d(384)
        
        # 第四组深度可分离卷积：增大通道数、减小特征图尺寸
        self.dwconv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=384)
        self.pwconv4 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=1, stride=2, padding=0)
        self.bn_dw4 = nn.BatchNorm2d(768)
        
        # 全连接层
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(768*1*1, 96)
        self.bn_fc1 = nn.BatchNorm1d(96)
        self.fc2 = nn.Linear(96, num_classes)
        
    def forward(self, x):

        # # 第一组1×1卷积
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = F.relu(x)

        # 第一组深度可分离卷积
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self.bn_dw1(x)
        x = F.relu(x)
        
        # 第二组深度可分离卷积
        x = self.dwconv2(x)
        x = self.pwconv2(x)
        x = self.bn_dw2(x)
        x = F.relu(x)
        
        # 第三组深度可分离卷积
        x = self.dwconv3(x)
        x = self.pwconv3(x)
        x = self.bn_dw3(x)
        x = F.relu(x)
        
        # 第四组深度可分离卷积
        x = self.dwconv4(x)
        x = self.pwconv4(x)
        x = self.bn_dw4(x)
        x = F.relu(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x) 

        x = self.dropout(x)
        x = self.fc2(x)

        return x

MelHilbertTimeLDSCNN = MelHilbertTimeCNN

'''
4. 单通道Hilbert图像CNN
'''
class SignalHilbertCNN(nn.Module):
    def __init__(self, num_classes=3, hilbert_height=128, hilbert_width=128, **kwargs):
        super(SignalHilbertCNN, self).__init__()
        
        # 第一个深度可分离卷积-BN组合
        self.dwconv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, groups=1)
        self.pwconv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        
        # 第二个深度可分离卷积-BN组合
        self.dwconv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8)
        self.pwconv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 第三个深度可分离卷积-BN组合
        self.dwconv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, groups=16)
        self.pwconv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        
        # 第四个深度可分离卷积-BN组合
        self.dwconv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32)
        self.pwconv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 第五个深度可分离卷积-BN组合
        self.dwconv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pwconv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(128)
        
        # 第六个深度可分离卷积-BN组合
        self.dwconv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.pwconv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(256)
        
        # 第七个深度可分离卷积-BN组合
        self.dwconv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256)
        self.pwconv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)
        self.bn7 = nn.BatchNorm2d(512)
        
        # 计算全连接层的输入维度
        # 经过7次下采样 (stride=2)，尺寸变为原来的 1/128
        h_out = hilbert_height // 128
        w_out = hilbert_width // 128
        fc_input_dim = 512 * h_out * w_out
        
        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 深度可分离卷积-BN层
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.dwconv2(x)
        x = self.pwconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.dwconv3(x)
        x = self.pwconv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.dwconv4(x)
        x = self.pwconv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.dwconv5(x)
        x = self.pwconv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        x = self.dwconv6(x)
        x = self.pwconv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        
        x = self.dwconv7(x)
        x = self.pwconv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        # 展平操作
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
SignalHilbertLDSCNN = SignalHilbertCNN

'''
5. 1D Signal CNN
'''
class SignalCNN(nn.Module):
    def __init__(self, num_classes=3, signal_length=1024, **kwargs):
        super(SignalCNN, self).__init__()
        
        self.dwconv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, groups=1)
        self.pwconv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1, stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(8)
        
        self.dwconv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8)
        self.pwconv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=2, padding=0)
        self.bn2 = nn.BatchNorm1d(16)
        
        self.dwconv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1, groups=16)
        self.pwconv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.dwconv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32)
        self.pwconv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.dwconv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pwconv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.dwconv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64)
        self.pwconv6 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.bn6 = nn.BatchNorm1d(128)
        
        self.dwconv7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.pwconv7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.bn7 = nn.BatchNorm1d(128)
        
        self.dwconv8 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128)
        self.pwconv8 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)
        self.bn8 = nn.BatchNorm1d(256)
        
        self.dwconv9 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256)
        self.pwconv9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)
        self.bn9 = nn.BatchNorm1d(512)
        
        self.dwconv10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1, groups=512)
        self.pwconv10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=2, padding=0)
        self.bn10 = nn.BatchNorm1d(512)
        
        # 计算全连接层的输入维度
        # 经过7次下采样 (stride=2)，尺寸变为原来的 1/128
        h_out = signal_length // 1024
        w_out = 1
        fc_input_dim = 512 * h_out * w_out
        
        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 深度可分离卷积-BN层
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.dwconv2(x)
        x = self.pwconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.dwconv3(x)
        x = self.pwconv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.dwconv4(x)
        x = self.pwconv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.dwconv5(x)
        x = self.pwconv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        x = self.dwconv6(x)
        x = self.pwconv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        
        x = self.dwconv7(x)
        x = self.pwconv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        x = self.dwconv8(x)
        x = self.pwconv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        
        x = self.dwconv9(x)
        x = self.pwconv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        
        x = self.dwconv10(x)
        x = self.pwconv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        
        # 展平操作
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

SignalLDSCNN = SignalCNN

# if __name__ == "__main__":
    
#     import torch
#     from torchsummary import summary

#     model = SignalCNN().to("cuda")
#     summary(model, (1, 1024))