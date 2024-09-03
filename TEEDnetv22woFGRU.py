import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ted import TED


# DA模块
class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int, c_m: int, c_n: int, reconstruct=False):
        super(DoubleAttentionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size=1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size=1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size=1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        batch_size, c, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        x = self.bn(x)
        A = self.convA(x)  # (B, c_m, h, w)
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.view(batch_size, self.c_m, h * w)
        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim=-1)
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        attention_vectors = F.softmax(attention_vectors, dim=1)  # (B, c_n, h * w)
        tmpZ = global_descriptors.matmul(attention_vectors)  # (B, c_m, h * w)
        tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ


class HarmonicBlock(nn.Module):
    def __init__(self, input_channels, kernel_sizes):
        super(HarmonicBlock, self).__init__()
        
        # Conv2D layers before entering the main blocks
        self.x_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 2 * input_channels, (3, 3), padding=1),
            nn.ReLU()
        )
        self.x_conv2 = nn.Sequential(
            nn.Conv2d(2 * input_channels, 3 * input_channels, (3, 3), padding=1),
            nn.ReLU()
        )
        
        # Temporal branch
        self.avg_pool_temporal = nn.AdaptiveAvgPool2d((1, None))  # Pool across frequency axis
        
        self.conv1d_temporal = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_channels, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Frequency (Harmonic) branch with customizable kernel sizes
        self.avg_pool_harmonic = nn.AdaptiveAvgPool2d((None, 1))  # Pool across time axis
        
        self.conv1d_large = nn.Sequential(
                nn.Conv1d(input_channels, input_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
                nn.ReLU(),
                nn.Conv1d(input_channels, input_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
                nn.Sigmoid()
        )
        self.conv1d_mid = nn.Sequential(
                nn.Conv1d(input_channels, input_channels, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
                nn.ReLU(),
                nn.Conv1d(input_channels, input_channels, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
                nn.Sigmoid()
        )
        self.conv1d_tiny = nn.Sequential(
                nn.Conv1d(input_channels, input_channels, kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2),
                nn.ReLU(),
                nn.Conv1d(input_channels, input_channels, kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2),
                nn.Sigmoid()
        )
        
        # Final Conv2D after multiplication and addition (3C -> C)
        self.final_conv = nn.Conv2d(4*input_channels, input_channels, kernel_size=1)
    
    
    def forward(self, x):
        B, C, Fre, T = x.size()
        
        # Initial Conv2D
        input = self.x_conv1(x)
        input = self.x_conv2(input)  # (B, C, F, T)
        
        # Temporal path
        x_temp = self.avg_pool_temporal(x)  # (B, C, 1, T)
        x_temp = x_temp.squeeze(2)  # (B, C, T)
        x_temp = self.conv1d_temporal(x_temp)  # (B, C, T)
        #x_temp = self.softmax_temporal(x_temp)  # (B, C, T)
        x_temp = x_temp.unsqueeze(2)  # (B, C, 1, T)
        
        # Frequency (Harmonic) path
        x_freq = self.avg_pool_harmonic(x)  # (B, C, F, 1)
        #x_freq = x.view(B, C, Fre * T) # (B, C, F*T)
        x_freq = x_freq.squeeze(3)  # (B, C, F)
        
        # Apply different Conv1D layers in parallel with specified kernel sizes
        x_freq_large = self.conv1d_large(x_freq)  # (B, C, F*T)
        x_freq_mid = self.conv1d_mid(x_freq)  # (B, C, F*T)
        x_freq_tiny = self.conv1d_tiny(x_freq)  # (B, C, F*T)
        
        # Softmax on each output
        # x_freq_large = x_freq_large[:, :, :Fre * T]  # (B, C, F*T)
        # x_freq_mid = x_freq_mid[:, :, :Fre * T]  # (B, C, F*T)
        # x_freq_tiny = x_freq_tiny[:, :, :Fre * T] # (B, C, F*T)

        x_freq_large = x_freq_large[:, :, :Fre]  # (B, C, F*T)
        x_freq_mid = x_freq_mid[:, :, :Fre]  # (B, C, F*T)
        x_freq_tiny = x_freq_tiny[:, :, :Fre] # (B, C, F*T)

        
        # Concatenate the results from the frequency branch
        x_freq_concat = torch.cat((x_freq_large, x_freq_mid, x_freq_tiny), dim=1)  # (B, 3C, F*T)
        x_freq_concat = x_freq_concat.unsqueeze(3)
        #x_freq_concat = x_freq_concat.view(B, -1, Fre, T)
        
        # Repeat x_temp across the C dimension
        x_temp = x_temp.repeat(1, 3, 1, 1)  # (B, 3C, 1, T)
        
        # Multiply Temporal and Frequency branches
        x_mult = x_freq_concat * x_temp  # (B, 3C, F, T)
        
        
        # Sum Temporal and Frequency branches
        output = x_mult * input  # (B, 3C, F, T)
        
        # Final Conv2D layer (4C -> C)
        output = self.final_conv(torch.cat([output, x],dim= 1))  # (B, C, F, T)
        
        return output

# UpSample模块
class UpSample(nn.Module):
    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)

class Downsampling(nn.Module):
    def __init__(self, input_channels, emb_channels, output_channels, kernel_sizes_full, kernel_sizes_mid, kernel_sizes_tiny):
        super(Downsampling, self).__init__()
        self.ted = TED(inputchannels=input_channels, outputchannels=emb_channels)
        
        self.hb_full = HarmonicBlock(input_channels=emb_channels, kernel_sizes=kernel_sizes_full)
        self.hb_mid = HarmonicBlock(input_channels=emb_channels,  kernel_sizes=kernel_sizes_mid)
        self.hb_tiny = HarmonicBlock(input_channels=emb_channels,  kernel_sizes=kernel_sizes_tiny)

        self.da = DoubleAttentionLayer(in_channels=4 * emb_channels, c_m=output_channels, c_n=emb_channels // 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.tfa = TFatten(inputchannels=output_channels, outputchannels=output_channels)

    def forward(self, input):
        x = self.ted(input)  # (B, C_emb, F, T)
        
        x_full = self.hb_full(x)
        x_mid = self.hb_mid(x)
        x_tiny = self.hb_tiny(x)
        #x_combined = torch.cat([x_full, x_mid, x_tiny], dim=1)  # (B, 2*C_emb, F, T)

        x_tfa_combined = torch.cat([x_full, x_mid, x_tiny, x], dim=1)  # (B, 3*C_emb, F, T)
        x_out = self.da(x_tfa_combined)  # (B, C_out, F, T)
        
        #x_out,_,_ = self.tfa(x_out)
        
        return self.pool(x_out), x_out, input


# Upsampling模块
class Upsampling(nn.Module):
    def __init__(self, input_channels, emb_channels, output_channels, kernel_sizes_full, kernel_sizes_mid, kernel_sizes_tiny):
        super(Upsampling, self).__init__()
        # self.da_input = DoubleAttentionLayer(in_channels=input_channels, c_m=emb_channels, c_n=emb_channels)
        
        self.ted = TED(inputchannels=input_channels, outputchannels=emb_channels)

        self.hb_full = HarmonicBlock(input_channels=emb_channels, kernel_sizes=kernel_sizes_full)
        self.hb_mid = HarmonicBlock(input_channels=emb_channels, kernel_sizes=kernel_sizes_mid)
        self.hb_tiny = HarmonicBlock(input_channels=emb_channels, kernel_sizes=kernel_sizes_tiny)
        
        self.da_output = DoubleAttentionLayer(in_channels=emb_channels*4, c_m = output_channels, c_n=emb_channels // 2)
        self.upsample = UpSample(n_chan = output_channels, factor=2)
        #self.tfa = TFatten(inputchannels=output_channels, outputchannels=output_channels)

    def forward(self, x):
        x = self.ted(x)  # (B, C_emb, T, F)
        
        x_full = self.hb_full(x)
        x_mid = self.hb_mid(x)
        x_tiny = self.hb_tiny(x)
        
        x_out = self.da_output(torch.cat([x_full, x_mid, x_tiny, x], dim=1))  # (B, C_emb, T, F)

        x_out = self.upsample(x_out)  # (B, C_emb, T*2, F*2)

        #x_out,_,_ = self.tfa(x_out)
        
        return x_out
    
class GRUAlongFrequency(nn.Module):
    def __init__(self, Time, hidden_size, num_layers=1):
        super(GRUAlongFrequency, self).__init__()
        #self.conv1x1 = nn.Conv2d(in_channels, c_m, kernel_size=1)

    def forward(self, x):
        B, C, F, T = x.size()

        return x

class GRUAlongTime(nn.Module):
    def __init__(self, Frequency, hidden_size, num_layers=1):
        super(GRUAlongTime, self).__init__()
        self.gru = nn.GRU(input_size=Frequency, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, Frequency)

    def forward(self, x):
        B, C, F, T = x.size()
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, T, F)
        x = x.view(B * C, T, F)  # (B*C, T, F)
        output, _ = self.gru(x)  # (B*C, T, hidden_size)
        output = self.proj(output)  # (B*C, T, F)
        output = output.view(B, C, T, F)  # (B, C, T, F)
        output = output.permute(0, 1, 3, 2).contiguous()  # (B, C, F, T)
        return output

class GRUBottleneck(nn.Module):
    def __init__(self, Time, Frequency, hidden_size, num_layers=1):
        super(GRUBottleneck, self).__init__()
        self.gru_time = GRUAlongTime(Frequency, hidden_size, num_layers)
        self.gru_frequency = GRUAlongFrequency(Time, hidden_size, num_layers)
        #self.proj = nn.Conv2d(2 * input_channels, input_channels, kernel_size=1)

    def forward(self, x):
        output_time = self.gru_time(x)
        output_frequency = self.gru_frequency(x)
        output_combined = torch.cat([output_time, output_frequency], dim=1)  # (B, 2*C, T, F)
        #output = self.proj(output_combined)  # (B, C, T, F)
        return output_combined


# 主网络架构
class HAUnetv2woFGRU(nn.Module):
    def __init__(self, Time=128, Frequency=360):
        super(HAUnetv2woFGRU, self).__init__()

        
        self.x_down_1 = Downsampling(input_channels=3, 
                                emb_channels=16, 
                                output_channels=32,
                                kernel_sizes_full = [60, 210, 360],
                                kernel_sizes_mid = [120, 150, 240],
                                kernel_sizes_tiny = [60, 90, 120]
                                )
        
        self.x_down_2 = Downsampling(input_channels=32, 
                                emb_channels=32, 
                                output_channels=64,
                                kernel_sizes_full = [30, 105, 180],
                                kernel_sizes_mid = [30, 75, 120],
                                kernel_sizes_tiny = [30, 45, 60]
                                )
        
        self.x_down_3 = Downsampling(input_channels=64, 
                                emb_channels=64, 
                                output_channels=128,
                                kernel_sizes_full = [15, 55, 90],
                                kernel_sizes_mid = [15, 40, 60],
                                kernel_sizes_tiny = [15, 25, 30]
                                )

        self.x_neck = GRUBottleneck(Time = Time//8, Frequency= Frequency//8,hidden_size=64)
        
        self.x_up_3 = Upsampling(input_channels=128*3, 
                                emb_channels=128, 
                                output_channels=128,
                                kernel_sizes_full = [60, 210, 360],
                                kernel_sizes_mid = [120, 210, 300],
                                kernel_sizes_tiny = [180, 210, 240]
                                )
        self.x_up_2 = Upsampling(input_channels=128*3, 
                                emb_channels=64, 
                                output_channels=64,
                                kernel_sizes_full = [30, 105, 180],
                                kernel_sizes_mid = [60, 105, 150],
                                kernel_sizes_tiny = [90, 105, 120]
                                )
        self.x_up_1 = Upsampling(input_channels=64*3, 
                                emb_channels=32, 
                                output_channels=32,
                                kernel_sizes_full = [50, 150, 300],
                                kernel_sizes_mid = [40, 80, 160],
                                kernel_sizes_tiny = [30, 45, 60]
                                )
        self.x_skip_1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.SELU(),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.SELU()
        )
        self.x_skip_2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.SELU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.SELU()
        )
        self.x_skip_3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.SELU(),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.SELU()
        )
        
        self.x_out = nn.Sequential(
            nn.Conv2d(32*3, 32, 5, padding=2),
            nn.SELU(),
            #nn.BatchNorm2d(16),
            nn.Conv2d(32, 1, 5, padding=2),
            nn.SELU()
        )
        
        self.bm_layer = nn.Sequential(
            nn.Conv2d(3, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (3, 1), stride=(3, 1)),#3
            nn.SELU(),
            nn.Conv2d(16, 16, (6, 1), stride=(6, 1)),#6
            nn.SELU(),
            nn.Conv2d(16, 1, (5, 1), stride=(5, 1)),
            nn.SELU()
        )
        self.bn_layer = nn.Sequential(
            nn.BatchNorm2d(32*3),
            nn.Conv2d(32*3, 32, 5, padding=2),
            nn.SELU(),
            #nn.BatchNorm2d(16),
            nn.Conv2d(32, 3, 5, padding=2),
            nn.SELU(),
            nn.BatchNorm2d(3)
        )

    def forward(self, input):
        # (B, 3, T, F)
        B, _, F, T = input.size()
        
        # Downsampling path
        x1, x1_out, x1_orig = self.x_down_1(input)
        x2, x2_out, x2_orig = self.x_down_2(x1)
        x3, x3_out, x3_orig = self.x_down_3(x2)

        # Bottleneck
        x_neck = self.x_neck(x3)

        # Upsampling path
        x_up_3 = self.x_up_3(torch.cat([x_neck, x3], dim=1))
        x_up_2 = self.x_up_2(torch.cat([x_up_3, x3_out, self.x_skip_3(x3_orig)], dim=1))
        x_up_1 = self.x_up_1(torch.cat([x_up_2, x2_out, self.x_skip_2(x2_orig)], dim=1))
        
        x_out_combine = torch.cat([x_up_1, x1_out,self.x_skip_1(x1_orig)], dim=1)
        
        # Output layer
        x_out = self.x_out(x_out_combine)
        
        bm = self.bn_layer(x_out_combine)
        bm = self.bm_layer(bm)  # (b,1,1,128)
        
        output_pre = torch.cat([bm, x_out], dim=2)
        output = nn.Softmax(dim=-2)(output_pre)
        
        

        return output, output_pre

