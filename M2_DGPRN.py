import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

unet_channel_list = [16, 32, 64, 128, 256, 512, 1024]
gcn_channel_list = [512, 256, 128, 64, 32, 8, 4]

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, strides = 1, padding = 1):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, padding = padding, bias = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = strides, padding = padding, bias = True)
        self.conv3 = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv3(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, strides = 2):
        super(Deconv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, bias = True)
        self.conv2 = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.layer1_conv = Conv(11, unet_channel_list[0])
        self.layer2_conv = Conv(unet_channel_list[0], unet_channel_list[1])
        self.layer3_conv = Conv(unet_channel_list[1], unet_channel_list[2])
        self.layer4_conv = Conv(unet_channel_list[2], unet_channel_list[3])
        self.layer5_conv = Conv(unet_channel_list[3], unet_channel_list[4])
        self.layer6_conv = Conv(unet_channel_list[4], unet_channel_list[5])
        self.layer7_conv = Conv(unet_channel_list[5], unet_channel_list[6])
        self.layer8_conv = Conv(unet_channel_list[6], unet_channel_list[5])
        self.layer9_conv = Conv(unet_channel_list[5], unet_channel_list[4])
        self.layer10_conv = Conv(unet_channel_list[4], unet_channel_list[3])
        self.layer11_conv = Conv(unet_channel_list[3], unet_channel_list[2])
        self.layer12_conv = Conv(unet_channel_list[2], unet_channel_list[1])
        self.deconv1 = Deconv(unet_channel_list[6], unet_channel_list[5])
        self.deconv2 = Deconv(unet_channel_list[5], unet_channel_list[4])
        self.deconv3 = Deconv(unet_channel_list[4], unet_channel_list[3])
        self.deconv4 = Deconv(unet_channel_list[3], unet_channel_list[2])
        self.deconv5 = Deconv(unet_channel_list[2], unet_channel_list[1])
        self.deconv6 = Deconv(unet_channel_list[1], unet_channel_list[0])
        self.layer_last = nn.Conv2d(unet_channel_list[1], unet_channel_list[2], kernel_size = 3, stride = 1, padding = 1, bias = True)
    def forward(self, x, mask_net, lat_len, lon_len):
        x = x.reshape([-1, lat_len * 8, lon_len * 8, 11])
        x = x.permute(0, 3, 1, 2)
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)
        conv5 = self.layer5_conv(pool4)
        pool5 = F.max_pool2d(conv5, 2)
        conv6 = self.layer6_conv(pool5)
        pool6 = F.max_pool2d(conv6, 2)
        conv7 = self.layer7_conv(pool6)
        deconv1 = self.deconv1(conv7)
        concat1 = torch.cat([conv6, deconv1], dim = 1)
        conv8 = self.layer8_conv(concat1)
        deconv2 = self.deconv2(conv8)
        concat2 = torch.cat([conv5, deconv2], dim = 1)
        conv9 = self.layer9_conv(concat2)
        deconv3 = self.deconv3(conv9)
        concat3 = torch.cat([conv4, deconv3], dim = 1)
        conv10 = self.layer10_conv(concat3)
        deconv4 = self.deconv4(conv10)
        concat4 = torch.cat([conv3, deconv4], dim = 1)
        conv11 = self.layer11_conv(concat4)
        deconv5 = self.deconv5(conv11)
        concat5 = torch.cat([conv2, deconv5], dim = 1)
        conv12 = self.layer12_conv(concat5)
        deconv6 = self.deconv6(conv12)
        concat6 = torch.cat([conv1, deconv6], dim = 1)
        out = self.layer_last(concat6)
        out = out.permute(0, 2, 3, 1)
        out = out.reshape([-1, lat_len * 8 * lon_len * 8, unet_channel_list[2]])
        out[mask_net] = 0
        return out

class Conv_Res1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, strides = 1, padding = 1):
        super(Conv_Res1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, padding = padding, bias = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = strides, padding = padding, bias = True)
        self.conv3 = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv3(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class Deconv_Res1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, strides = 2):
        super(Deconv_Res1, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, bias = True)
        self.conv2 = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class UNet_Res1(nn.Module):
    def __init__(self):
        super(UNet_Res1, self).__init__()
        self.linear_e1_1 = torch.nn.Linear(4, 64)
        self.linear_e1_2 = torch.nn.Linear(64, 128)
        self.linear_e1_3 = torch.nn.Linear(128, 64)
        self.linear_e1_4 = torch.nn.Linear(64, 10)
        self.linear_e2_1 = torch.nn.Linear(4, 64)
        self.linear_e2_2 = torch.nn.Linear(64, 128)
        self.linear_e2_3 = torch.nn.Linear(128, 64)
        self.linear_e2_4 = torch.nn.Linear(64, 10)
        self.layer1_conv = Conv(11, unet_channel_list[0])
        self.layer2_conv = Conv(unet_channel_list[0], unet_channel_list[1])
        self.layer3_conv = Conv(unet_channel_list[1], unet_channel_list[2])
        self.layer4_conv = Conv(unet_channel_list[2], unet_channel_list[3])
        self.layer5_conv = Conv(unet_channel_list[3], unet_channel_list[4])
        self.layer6_conv = Conv(unet_channel_list[4], unet_channel_list[5])
        self.layer7_conv = Conv(unet_channel_list[5], unet_channel_list[4])
        self.layer8_conv = Conv(unet_channel_list[4], unet_channel_list[3])
        self.layer9_conv = Conv(unet_channel_list[3], unet_channel_list[2])
        self.layer10_conv = Conv(unet_channel_list[2], unet_channel_list[1])
        self.deconv1 = Deconv(unet_channel_list[5], unet_channel_list[4])
        self.deconv2 = Deconv(unet_channel_list[4], unet_channel_list[3])
        self.deconv3 = Deconv(unet_channel_list[3], unet_channel_list[2])
        self.deconv4 = Deconv(unet_channel_list[2], unet_channel_list[1])
        self.deconv5 = Deconv(unet_channel_list[1], unet_channel_list[0])
        self.layer_last = nn.Conv2d(unet_channel_list[1], 4, kernel_size = 3, stride = 1, padding = 1, bias = True)
    def forward(self, x, topography, saa, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std):
        mask_loss = torch.where(mask_loss == 0)
        x[mask_loss] = 0
        topography = topography.reshape([-1, lat_len * 8, lon_len * 8])
        saa = saa.reshape([-1, lat_len * 8, lon_len * 8])
        topography = F.max_pool2d(topography, kernel_size = 2)
        saa = F.max_pool2d(saa, kernel_size = 2)
        saa = torch.flatten(saa, start_dim = 1, end_dim = 2)
        ssws = x[:, :, 0]
        ussw = x[:, :, 1]
        vssw = x[:, :, 2]
        sst = x[:, :, 3]

        e1 = self.linear_e1_1(x)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_2(e1)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_3(e1)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_4(e1)
        e2 = self.linear_e2_1(x)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_2(e2)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_3(e2)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_4(e2)

        sswd = 180 + torch.arctan2(ussw * sswsst_std[1] + sswsst_mean[1], vssw * sswsst_std[2] + sswsst_mean[2]) * (180 / torch.pi)

        relative_dirwind = torch.abs(saa - sswd)
        relative_dirwind = torch.nan_to_num(relative_dirwind, nan = 0)
        adjust_sswd_idx = torch.where(relative_dirwind >= 180)
        relative_dirwind[adjust_sswd_idx] = 360 - relative_dirwind[adjust_sswd_idx]
        relative_dirwind = relative_dirwind * torch.pi / 180

        aod = 0.15 / (1 + 6.7 * torch.exp(-0.17 * (ssws * sswsst_std[0] + sswsst_mean[0])))
        res = ((e1.permute(2, 0, 1) * torch.cos(relative_dirwind) + e2.permute(2, 0, 1) * torch.cos(2 * relative_dirwind)) * (aod ** 2) * sst).permute(1, 2, 0)
        res = res.reshape([-1, lat_len * 4, lon_len * 4, 10])
        res = torch.cat([res, topography.unsqueeze(dim = 3)], dim = 3)
        res = res.permute(0, 3, 1, 2)
        conv1 = self.layer1_conv(res)
        pool1 = F.max_pool2d(conv1, 2)
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)
        conv5 = self.layer5_conv(pool4)
        pool5 = F.max_pool2d(conv5, 2)
        conv6 = self.layer6_conv(pool5)
        deconv1 = self.deconv1(conv6)
        concat1 = torch.cat([conv5, deconv1], dim = 1)
        conv7 = self.layer7_conv(concat1)
        deconv2 = self.deconv2(conv7)
        concat2 = torch.cat([conv4, deconv2], dim = 1)
        conv8 = self.layer8_conv(concat2)
        deconv3 = self.deconv3(conv8)
        concat3 = torch.cat([conv3, deconv3], dim = 1)
        conv9 = self.layer9_conv(concat3)
        deconv4 = self.deconv4(conv9)
        concat4 = torch.cat([conv2, deconv4], dim = 1)
        conv10 = self.layer10_conv(concat4)
        deconv5 = self.deconv5(conv10)
        concat5 = torch.cat([conv1, deconv5], dim = 1)
        res = self.layer_last(concat5)
        res = res.permute(0, 2, 3, 1)
        res = res.reshape([-1, lat_len * 4 * lon_len * 4, 4])
        res[mask_loss] = 0
        output = x + res
        return output

class Conv_Res2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, strides = 1, padding = 1):
        super(Conv_Res2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, padding = padding, bias = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = strides, padding = padding, bias = True)
        self.conv3 = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv3(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class Deconv_Res2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, strides = 2):
        super(Deconv_Res2, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, bias = True)
        self.conv2 = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class UNet_Res2(nn.Module):
    def __init__(self):
        super(UNet_Res2, self).__init__()
        self.linear_e1_1 = torch.nn.Linear(4, 64)
        self.linear_e1_2 = torch.nn.Linear(64, 128)
        self.linear_e1_3 = torch.nn.Linear(128, 64)
        self.linear_e1_4 = torch.nn.Linear(64, 10)
        self.linear_e2_1 = torch.nn.Linear(4, 64)
        self.linear_e2_2 = torch.nn.Linear(64, 128)
        self.linear_e2_3 = torch.nn.Linear(128, 64)
        self.linear_e2_4 = torch.nn.Linear(64, 10)
        self.layer1_conv = Conv(11, unet_channel_list[0])
        self.layer2_conv = Conv(unet_channel_list[0], unet_channel_list[1])
        self.layer3_conv = Conv(unet_channel_list[1], unet_channel_list[2])
        self.layer4_conv = Conv(unet_channel_list[2], unet_channel_list[3])
        self.layer5_conv = Conv(unet_channel_list[3], unet_channel_list[4])
        self.layer6_conv = Conv(unet_channel_list[4], unet_channel_list[5])
        self.layer7_conv = Conv(unet_channel_list[5], unet_channel_list[4])
        self.layer8_conv = Conv(unet_channel_list[4], unet_channel_list[3])
        self.layer9_conv = Conv(unet_channel_list[3], unet_channel_list[2])
        self.layer10_conv = Conv(unet_channel_list[2], unet_channel_list[1])
        self.deconv1 = Deconv(unet_channel_list[5], unet_channel_list[4])
        self.deconv2 = Deconv(unet_channel_list[4], unet_channel_list[3])
        self.deconv3 = Deconv(unet_channel_list[3], unet_channel_list[2])
        self.deconv4 = Deconv(unet_channel_list[2], unet_channel_list[1])
        self.deconv5 = Deconv(unet_channel_list[1], unet_channel_list[0])
        self.layer_last = nn.Conv2d(unet_channel_list[1], 4, kernel_size = 3, stride = 1, padding = 1, bias = True)
    def forward(self, x, topography, saa, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std):
        mask_loss = torch.where(mask_loss == 0)
        x[mask_loss] = 0
        topography = topography.reshape([-1, lat_len * 8, lon_len * 8])
        saa = saa.reshape([-1, lat_len * 8, lon_len * 8])
        topography = F.max_pool2d(topography, kernel_size = 2)
        saa = F.max_pool2d(saa, kernel_size = 2)
        saa = torch.flatten(saa, start_dim = 1, end_dim = 2)
        ssws = x[:, :, 0]
        ussw = x[:, :, 1]
        vssw = x[:, :, 2]
        sst = x[:, :, 3]

        e1 = self.linear_e1_1(x)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_2(e1)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_3(e1)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_4(e1)
        e2 = self.linear_e2_1(x)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_2(e2)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_3(e2)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_4(e2)

        sswd = 180 + torch.arctan2(ussw * sswsst_std[1] + sswsst_mean[1], vssw * sswsst_std[2] + sswsst_mean[2]) * (180 / torch.pi)

        relative_dirwind = torch.abs(saa - sswd)
        relative_dirwind = torch.nan_to_num(relative_dirwind, nan = 0)
        adjust_sswd_idx = torch.where(relative_dirwind >= 180)
        relative_dirwind[adjust_sswd_idx] = 360 - relative_dirwind[adjust_sswd_idx]
        relative_dirwind = relative_dirwind * torch.pi / 180

        aod = 0.15 / (1 + 6.7 * torch.exp(-0.17 * (ssws * sswsst_std[0] + sswsst_mean[0])))
        res = ((e1.permute(2, 0, 1) * torch.cos(relative_dirwind) + e2.permute(2, 0, 1) * torch.cos(2 * relative_dirwind)) * (aod ** 2) * sst).permute(1, 2, 0)
        res = res.reshape([-1, lat_len * 4, lon_len * 4, 10])
        res = torch.cat([res, topography.unsqueeze(dim = 3)], dim = 3)
        res = res.permute(0, 3, 1, 2)
        conv1 = self.layer1_conv(res)
        pool1 = F.max_pool2d(conv1, 2)
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)
        conv5 = self.layer5_conv(pool4)
        pool5 = F.max_pool2d(conv5, 2)
        conv6 = self.layer6_conv(pool5)
        deconv1 = self.deconv1(conv6)
        concat1 = torch.cat([conv5, deconv1], dim = 1)
        conv7 = self.layer7_conv(concat1)
        deconv2 = self.deconv2(conv7)
        concat2 = torch.cat([conv4, deconv2], dim = 1)
        conv8 = self.layer8_conv(concat2)
        deconv3 = self.deconv3(conv8)
        concat3 = torch.cat([conv3, deconv3], dim = 1)
        conv9 = self.layer9_conv(concat3)
        deconv4 = self.deconv4(conv9)
        concat4 = torch.cat([conv2, deconv4], dim = 1)
        conv10 = self.layer10_conv(concat4)
        deconv5 = self.deconv5(conv10)
        concat5 = torch.cat([conv1, deconv5], dim = 1)
        res = self.layer_last(concat5)
        res = res.permute(0, 2, 3, 1)
        res = res.reshape([-1, lat_len * 4 * lon_len * 4, 4])
        res[mask_loss] = 0
        output = x + res
        return output

class Conv_Res3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, strides = 1, padding = 1):
        super(Conv_Res3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, padding = padding, bias = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = strides, padding = padding, bias = True)
        self.conv3 = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv3(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

class Deconv_Res3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, strides = 2):
        super(Deconv_Res3, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = strides, bias = True)
        self.conv2 = nn.LeakyReLU(0.2)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class UNet_Res3(nn.Module):
    def __init__(self):
        super(UNet_Res3, self).__init__()
        self.linear_e1_1 = torch.nn.Linear(4, 64)
        self.linear_e1_2 = torch.nn.Linear(64, 128)
        self.linear_e1_3 = torch.nn.Linear(128, 64)
        self.linear_e1_4 = torch.nn.Linear(64, 10)
        self.linear_e2_1 = torch.nn.Linear(4, 64)
        self.linear_e2_2 = torch.nn.Linear(64, 128)
        self.linear_e2_3 = torch.nn.Linear(128, 64)
        self.linear_e2_4 = torch.nn.Linear(64, 10)
        self.layer1_conv = Conv(11, unet_channel_list[0])
        self.layer2_conv = Conv(unet_channel_list[0], unet_channel_list[1])
        self.layer3_conv = Conv(unet_channel_list[1], unet_channel_list[2])
        self.layer4_conv = Conv(unet_channel_list[2], unet_channel_list[3])
        self.layer5_conv = Conv(unet_channel_list[3], unet_channel_list[4])
        self.layer6_conv = Conv(unet_channel_list[4], unet_channel_list[5])
        self.layer7_conv = Conv(unet_channel_list[5], unet_channel_list[4])
        self.layer8_conv = Conv(unet_channel_list[4], unet_channel_list[3])
        self.layer9_conv = Conv(unet_channel_list[3], unet_channel_list[2])
        self.layer10_conv = Conv(unet_channel_list[2], unet_channel_list[1])
        self.deconv1 = Deconv(unet_channel_list[5], unet_channel_list[4])
        self.deconv2 = Deconv(unet_channel_list[4], unet_channel_list[3])
        self.deconv3 = Deconv(unet_channel_list[3], unet_channel_list[2])
        self.deconv4 = Deconv(unet_channel_list[2], unet_channel_list[1])
        self.deconv5 = Deconv(unet_channel_list[1], unet_channel_list[0])
        self.layer_last = nn.Conv2d(unet_channel_list[1], 4, kernel_size = 3, stride = 1, padding = 1, bias = True)
    def forward(self, x, topography, saa, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std):
        mask_loss = torch.where(mask_loss == 0)
        x[mask_loss] = 0
        topography = topography.reshape([-1, lat_len * 8, lon_len * 8])
        saa = saa.reshape([-1, lat_len * 8, lon_len * 8])
        topography = F.max_pool2d(topography, kernel_size = 2)
        saa = F.max_pool2d(saa, kernel_size = 2)
        saa = torch.flatten(saa, start_dim = 1, end_dim = 2)
        ssws = x[:, :, 0]
        ussw = x[:, :, 1]
        vssw = x[:, :, 2]
        sst = x[:, :, 3]

        e1 = self.linear_e1_1(x)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_2(e1)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_3(e1)
        e1 = F.leaky_relu(e1, 0.2)
        e1 = self.linear_e1_4(e1)
        e2 = self.linear_e2_1(x)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_2(e2)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_3(e2)
        e2 = F.leaky_relu(e2, 0.2)
        e2 = self.linear_e2_4(e2)

        sswd = 180 + torch.arctan2(ussw * sswsst_std[1] + sswsst_mean[1], vssw * sswsst_std[2] + sswsst_mean[2]) * (180 / torch.pi)

        relative_dirwind = torch.abs(saa - sswd)
        relative_dirwind = torch.nan_to_num(relative_dirwind, nan = 0)
        adjust_sswd_idx = torch.where(relative_dirwind >= 180)
        relative_dirwind[adjust_sswd_idx] = 360 - relative_dirwind[adjust_sswd_idx]
        relative_dirwind = relative_dirwind * torch.pi / 180

        aod = 0.15 / (1 + 6.7 * torch.exp(-0.17 * (ssws * sswsst_std[0] + sswsst_mean[0])))
        res = ((e1.permute(2, 0, 1) * torch.cos(relative_dirwind) + e2.permute(2, 0, 1) * torch.cos(2 * relative_dirwind)) * (aod ** 2) * sst).permute(1, 2, 0)
        res = res.reshape([-1, lat_len * 4, lon_len * 4, 10])
        res = torch.cat([res, topography.unsqueeze(dim = 3)], dim = 3)
        res = res.permute(0, 3, 1, 2)
        conv1 = self.layer1_conv(res)
        pool1 = F.max_pool2d(conv1, 2)
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)
        conv5 = self.layer5_conv(pool4)
        pool5 = F.max_pool2d(conv5, 2)
        conv6 = self.layer6_conv(pool5)
        deconv1 = self.deconv1(conv6)
        concat1 = torch.cat([conv5, deconv1], dim = 1)
        conv7 = self.layer7_conv(concat1)
        deconv2 = self.deconv2(conv7)
        concat2 = torch.cat([conv4, deconv2], dim = 1)
        conv8 = self.layer8_conv(concat2)
        deconv3 = self.deconv3(conv8)
        concat3 = torch.cat([conv3, deconv3], dim = 1)
        conv9 = self.layer9_conv(concat3)
        deconv4 = self.deconv4(conv9)
        concat4 = torch.cat([conv2, deconv4], dim = 1)
        conv10 = self.layer10_conv(concat4)
        deconv5 = self.deconv5(conv10)
        concat5 = torch.cat([conv1, deconv5], dim = 1)
        res = self.layer_last(concat5)
        res = res.permute(0, 2, 3, 1)
        res = res.reshape([-1, lat_len * 4 * lon_len * 4, 4])
        res[mask_loss] = 0
        output = x + res
        return output

class DGPRN(torch.nn.Module):
    def __init__(self):
        super(DGPRN, self).__init__()
        self.embedding = UNet()
        self.conv1 = SAGEConv(unet_channel_list[2], gcn_channel_list[0])
        self.bn1 = nn.BatchNorm1d(gcn_channel_list[0])
        self.linear1 = torch.nn.Linear(gcn_channel_list[0], gcn_channel_list[0])
        self.bn2 = nn.BatchNorm1d(gcn_channel_list[0])
        self.conv2 = SAGEConv(gcn_channel_list[0], gcn_channel_list[0])
        self.bn3 = nn.BatchNorm1d(gcn_channel_list[0])
        self.linear2 = torch.nn.Linear(gcn_channel_list[0], gcn_channel_list[0])
        self.bn4 = nn.BatchNorm1d(gcn_channel_list[0])
        self.conv3 = SAGEConv(gcn_channel_list[0], gcn_channel_list[0])
        self.bn5 = nn.BatchNorm1d(gcn_channel_list[0])
        self.linear3 = torch.nn.Linear(gcn_channel_list[0], gcn_channel_list[1])
        self.linear4 = torch.nn.Linear(gcn_channel_list[1], gcn_channel_list[2])
        self.linear5 = torch.nn.Linear(gcn_channel_list[2], gcn_channel_list[2])
        self.linear6 = torch.nn.Linear(gcn_channel_list[2], gcn_channel_list[3])
        self.linear7 = torch.nn.Linear(gcn_channel_list[3], gcn_channel_list[4])
        self.linear8 = torch.nn.Linear(gcn_channel_list[4], gcn_channel_list[5])
        self.linear9 = torch.nn.Linear(gcn_channel_list[5], gcn_channel_list[6])
        self.resnet1 = UNet_Res1()
        self.resnet2 = UNet_Res2()
        self.resnet3 = UNet_Res3()
    def forward(self, x, saa, edge_idx, mask_net, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std, device):
        topography = x[:, :, 10]
        mask_value = torch.where(mask_net != 0)
        mask_net = torch.where(mask_net == 0)
        x = self.embedding(x, mask_net, lat_len, lon_len)
        x_batch = torch.zeros(x.shape[0], x.shape[1], gcn_channel_list[0]).to(device)
        for batch_num in range(x.shape[0]):
            x_batch[batch_num] = self.conv1(x[batch_num], edge_idx[batch_num][:, :edge_idx[batch_num, -1, -1]])
        x_batch_mask_value = x_batch[mask_value]
        x_batch_mask_value = self.bn1(x_batch_mask_value)
        x_batch[mask_value] = x_batch_mask_value
        x = F.leaky_relu(x_batch, 0.2)
        x = self.linear1(x)
        x_mask_value = x[mask_value]
        x_mask_value = self.bn2(x_mask_value)
        x[mask_value] = x_mask_value
        x = F.leaky_relu(x, 0.2)
        x_batch = torch.zeros(x.shape).to(device)
        for batch_num in range(x.shape[0]):
            x_batch[batch_num] = self.conv2(x[batch_num], edge_idx[batch_num][:, :edge_idx[batch_num, -1, -1]])
        x_batch_mask_value = x_batch[mask_value]
        x_batch_mask_value = self.bn3(x_batch_mask_value)
        x_batch[mask_value] = x_batch_mask_value
        x = F.leaky_relu(x_batch, 0.2)
        x = self.linear2(x)
        x_mask_value = x[mask_value]
        x_mask_value = self.bn4(x_mask_value)
        x[mask_value] = x_mask_value
        x = F.leaky_relu(x, 0.2)
        x_batch = torch.zeros(x.shape).to(device)
        for batch_num in range(x.shape[0]):
            x_batch[batch_num] = self.conv3(x[batch_num], edge_idx[batch_num][:, :edge_idx[batch_num, -1, -1]])
        x_batch_mask_value = x_batch[mask_value]
        x_batch_mask_value = self.bn5(x_batch_mask_value)
        x_batch[mask_value] = x_batch_mask_value
        x = F.leaky_relu(x_batch, 0.2)
        x = self.linear3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.linear4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.linear5(x)
        x = F.leaky_relu(x, 0.2)
        x = self.linear6(x)
        x = F.leaky_relu(x, 0.2)
        x = self.linear7(x)
        x = F.leaky_relu(x, 0.2)
        x = self.linear8(x)
        x = F.leaky_relu(x, 0.2)
        x = self.linear9(x)
        x[mask_net] = -99
        x = x.reshape([-1, lat_len * 8, lon_len * 8, 4]).permute(0, 3, 1, 2)
        x = F.max_pool2d(x, kernel_size = 2).permute(0, 2, 3, 1)
        output = torch.flatten(x, start_dim = 1, end_dim = 2)
        mask_pool = torch.where(output == -99)
        mask_loss[mask_pool] = False
        output = self.resnet1(output, topography, saa, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std)
        output = self.resnet2(output, topography, saa, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std)
        output = self.resnet3(output, topography, saa, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std)
        return output, mask_loss