import copy

import torch
from torch import nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU
from torch.nn import TransformerEncoder, TransformerEncoderLayer

WAVE_FEATURES = ["wave", "fft"]
SPEC_FEATURES = ["mel", "stft", "temp"]



class MFFC_2D(nn.Module):
    def __init__(self, channels=1, hidden_kernels=(3, 5, 7, 9)):
        super(MFFC_2D, self).__init__()
        self.channels = channels
        self.hidden_kernels = hidden_kernels
        self.bulit_model()

    def bulit_model(self):
        self.C_Relu_in = nn.Sequential(nn.Conv2d(in_channels=self.channels,
                                                 out_channels=self.channels,
                                                 kernel_size=(3, 3), stride=(1, 1),
                                                 padding=(1, 1), padding_mode="reflect"),
                                       nn.LeakyReLU())
        self.conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1))
        hidden_dict = {}
        for kernel in self.hidden_kernels:
            assert (kernel - 1) % 2 == 0, "kernel size should be 2 * x + 1"
            padding = (kernel - 1) // 2
            hidden_dict[str(kernel)+"ks"] = nn.Sequential(ComplexConv2d(in_channels=self.channels,
                                                                        out_channels=self.channels,
                                                                        kernel_size=(kernel, kernel),
                                                                        stride=(1, 1),
                                                                        padding=(padding, padding)),
                                                          ComplexReLU())
        self.hidden_modules = nn.ModuleDict(hidden_dict)

    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        assert len(x.size()) == 4, "wrong dim"
        x = self.C_Relu_in(x)
        x = self.FLF(x)
        x = self.conv(x)
        if x.size(1) == 1:
            x = x.squeeze(1)
        return x

    def FLF(self, x):
        x_ = torch.fft.rfft2(x)
        for i, kernel in enumerate(self.hidden_kernels):
            if i == 0:
                temp_ = self.hidden_modules[str(kernel) + "ks"](x_) #/ len(self.hidden_kernels)
            else:
                temp_ = temp_ + self.hidden_modules[str(kernel) + "ks"](x_) #/ len(self.hidden_kernels)
        x_ = temp_
        x = torch.fft.irfft2(x_, x.shape[-2:]) + x
        return x


class MFFC_1D(nn.Module):
    def __init__(self, channels=1, hidden_kernels=(3, 5, 7, 9)):
        super(MFFC_1D, self).__init__()
        self.channels = channels
        self.hidden_kernels = hidden_kernels
        self.bulit_model()

    def bulit_model(self):
        self.C_Relu_in = nn.Sequential(nn.Conv2d(in_channels=self.channels,
                                                 out_channels=self.channels,
                                                 kernel_size=(1, 3), stride=(1, 1),
                                                 padding=(0, 1), padding_mode="reflect"),
                                       nn.LeakyReLU())
        self.conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1))
        hidden_dict = {}
        for kernel in self.hidden_kernels:
            assert (kernel - 1) % 2 == 0, "kernel size should be 2 * x + 1"
            padding = (kernel - 1) // 2
            hidden_dict[str(kernel)+"ks"] = nn.Sequential(ComplexConv2d(in_channels=self.channels,
                                                                        out_channels=self.channels,
                                                                        kernel_size=(1, kernel),
                                                                        stride=(1, 1),
                                                                        padding=(1, padding)),
                                                          ComplexReLU())
        self.hidden_modules = nn.ModuleDict(hidden_dict)

    def forward(self, x):
        while len(x.size()) < 4:
            x = x.unsqueeze(1)
        assert len(x.size()) == 4, "wrong dim"
        x = self.C_Relu_in(x)
        x = self.FLF(x)
        x = self.conv(x)
        if x.size(1) == 1 and x.size(2) == 1:
            x = x.squeeze(1)
            x = x.squeeze(1)
        return x

    def FLF(self, x):
        x_ = torch.fft.rfft2(x)
        for i, kernel in enumerate(self.hidden_kernels):
            if i == 0:
                temp_ = self.hidden_modules[str(kernel) + "ks"](x_) #/ len(self.hidden_kernels)
            else:
                temp_ = temp_ + self.hidden_modules[str(kernel) + "ks"](x_) #/ len(self.hidden_kernels)
                # temp_ = temp_ + self.hidden_modules[str(kernel) + "ks"](x_)# + temp_
                # temp_ = copy.deepcopy(temp_)
        x_ = temp_
        x = torch.fft.irfft2(x_, x.shape[-2:]) + x
        return x


class SFB_SC_2D(nn.Module):
    def __init__(self, channels):
        super(SFB_SC_2D, self).__init__()
        self.channels = channels
        self.SC = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1), padding_mode="reflect"),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=channels, out_channels=channels,
                                          kernel_size=(3, 3), stride=(1, 1),
                                          padding=(1, 1), padding_mode="reflect")
                                )
    def forward(self, x):
        while len(x.size()) < 4:
            x = x.unsqueeze(1)
        return self.SC(x) + x


class SFB_SC_1D(nn.Module):
    def __init__(self, channels):
        super(SFB_SC_1D, self).__init__()
        self.channels = channels
        self.SC = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels,
                                          kernel_size=(1, 3), stride=(1, 1),
                                          padding=(0, 1), padding_mode="reflect"),
                                nn.LeakyReLU(),
                                nn.Conv2d(in_channels=channels, out_channels=channels,
                                          kernel_size=(1, 3), stride=(1, 1),
                                          padding=(0, 1), padding_mode="reflect")
                                )
    def forward(self, x):
        while len(x.size()) < 4:
            x = x.unsqueeze(1)
        return self.SC(x) + x


class CMFFC_2D(nn.Module):
    def __init__(self, channels, hidden_kernels):
        super(CMFFC_2D, self).__init__()
        self.channels = channels
        self.hidden_kernels = hidden_kernels
        self.SC = SFB_SC_2D(channels=channels)
        self.FFC = MFFC_2D(channels=channels, hidden_kernels=hidden_kernels)
        self.Conv = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=(1, 1))

    def forward(self, x):
        ffc_ = self.FFC(x)
        while len(ffc_.size()) < 4:
            ffc_ = ffc_.unsqueeze(1)
        x = torch.cat([self.SC(x), ffc_], dim=1)
        x = self.Conv(x)
        return x.squeeze(1)


class CMFFC_1D(nn.Module):
    def __init__(self, channels, hidden_kernels):
        super(CMFFC_1D, self).__init__()
        self.channels = channels
        self.hidden_kernels = hidden_kernels
        self.SC = SFB_SC_1D(channels=channels)
        self.FFC = MFFC_1D(channels=channels, hidden_kernels=hidden_kernels)
        self.Conv = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=(1, 1))
    def forward(self, x):
        ffc_ = self.FFC(x)
        while len(ffc_.size()) < 4:
            ffc_ = ffc_.unsqueeze(1)
        x = torch.cat([self.SC(x), ffc_], dim=1)
        x = self.Conv(x)
        return x.squeeze(1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=False), # TODO
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Resdual_DoubleConv(nn.Module):
    def __init__(self, nums=1, in_channels=1, mid_channels=None):
        super(Resdual_DoubleConv, self).__init__()
        self.nums = nums
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.mid_channels = mid_channels
        self.bulit_model()

    def forward(self, x):
        return self.blocks(x) + x

    def bulit_model(self):
        blocks = []
        for _ in range(self.nums):
            blocks.append(DoubleConv(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     mid_channels=self.mid_channels))
        self.blocks = nn.Sequential(*blocks)


class Resdual_DoubleConv_2D(nn.Module):
    def __init__(self, nums=1, in_channels=1, mid_channels=None):
        super(Resdual_DoubleConv_2D, self).__init__()
        self.nums = nums
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.mid_channels = mid_channels
        self.bulit_model()

    def forward(self, x):
        return self.blocks(x) + x

    def bulit_model(self):
        blocks = []
        for _ in range(self.nums):
            blocks.append(DoubleConv2D(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     mid_channels=self.mid_channels))
        self.blocks = nn.Sequential(*blocks)


class spec_encoder(nn.Module):
    def __init__(self, res_nums=1, channels=1, hidden_kernels=(3, 5, 7, 9),
                 resolution=(128, 138), hidden_features=256, single_mffc=True):
        super(spec_encoder, self).__init__()
        self.res_nums = res_nums
        self.channels = channels
        self.hidden_kernels = hidden_kernels
        self.resolution = resolution
        self.hidden_features = hidden_features
        self.single_mffc = single_mffc
        self.bulit_model()

    def bulit_model(self):
        self.Norm = nn.LayerNorm(self.resolution)
        # self.Res = Resdual_DoubleConv_2D(nums=self.res_nums, in_channels=self.channels)
        if self.single_mffc:
            self.Mffc = MFFC_2D(channels=self.channels, hidden_kernels=self.hidden_kernels)
        else:
            self.Mffc = CMFFC_2D(channels=self.channels, hidden_kernels=self.hidden_kernels)
        features = 1
        for r in self.resolution:
            features *= r
        self.Mlp =  nn.Linear(in_features=features, out_features=self.hidden_features)
        # Mlp(in_features=features, out_features=self.hidden_features)

    def forward(self, x):
        B = x.shape[0]
        x = self.Norm(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # x = self.Res(x)
        x = self.Mffc(x)
        x = x.contiguous().view(B, -1)
        return self.Mlp(x)


class wave_encoder(nn.Module):
    def __init__(self, res_nums=1, channels=1, hidden_kernels=(3, 5, 7, 9),
                 resolution=(4410,), hidden_features=256, single_mffc=True):
        super(wave_encoder, self).__init__()
        self.res_nums = res_nums
        self.channels = channels
        self.hidden_kernels = hidden_kernels
        self.resolution = resolution
        self.hidden_features = hidden_features
        self.single_mffc = single_mffc
        self.bulit_model()

    def forward(self, x):
        B = x.shape[0]
        x = self.Norm(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # x = self.Res(x)
        x = self.Mffc(x)
        x = x.contiguous().view(B, -1)
        return self.Mlp(x)

    def bulit_model(self):
        self.Norm = nn.LayerNorm(self.resolution)
        # self.Res = Resdual_DoubleConv(nums=self.res_nums, in_channels=self.channels)
        if self.single_mffc:
            self.Mffc = MFFC_1D(channels=self.channels, hidden_kernels=self.hidden_kernels)
        else:
            self.Mffc = CMFFC_1D(channels=self.channels, hidden_kernels=self.hidden_kernels)
        features = 1
        for r in self.resolution:
            features *= r
        self.Mlp = Mlp(in_features=features, out_features=self.hidden_features)
        # nn.Linear(in_features=features, out_features=self.hidden_features)
        #Mlp(in_features=features, out_features=self.hidden_features)



class fur_encoder(nn.Module):
    def __init__(self, res_nums, in_features, out_features):
        super(fur_encoder, self).__init__()
        self.res_nums = res_nums
        self.in_features = in_features
        self.out_features = out_features
        self.bulit_model()

    def forward(self, x):
        B = x.shape[0]
        while len(x.shape) < 3:
            x = x.unsqueeze(1)
        x = self.ResConv(x)
        x = x.contiguous().view(B, -1)
        x = self.Mlp(x)
        return x

    def bulit_model(self):
        self.ResConv = Resdual_DoubleConv(in_channels=1, nums=self.res_nums)
        self.Mlp = Mlp(in_features=self.in_features, out_features=self.out_features)
        # nn.Linear(in_features=self.in_features, out_features=self.out_features)
        # Mlp(in_features=self.in_features, out_features=self.out_features)


class classifier_v1(nn.Module):
    def __init__(self, in_channles, feature_dim, encoder_num, hidden_kernels=(3, 5, 7, 9),
                 res_nums=3, num_classes=4, single_mffc=True):
        super(classifier_v1, self).__init__()
        self.in_channles = in_channles
        self.feature_dim = feature_dim
        self.encoder_num = encoder_num
        self.hidden_kernels = hidden_kernels
        self.num_classes = num_classes
        self.res_nums = res_nums
        self.single_mffc = single_mffc
        self.bulit_model()

    def bulit_model(self):
        self.Norm = nn.LayerNorm([self.in_channles, self.feature_dim])
        encoder_layers_f = TransformerEncoderLayer(d_model=self.feature_dim,
                                                   dim_feedforward=2 * self.feature_dim, nhead=2, )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, self.encoder_num)
        # self.Conv = nn.Conv1d(in_channels=self.in_channles, out_channels=1, kernel_size=1)
        self.down = Mlp(in_features=self.in_channles, out_features=1)
        #nn.Linear(in_features=self.in_channles, out_features=1)
        # Mlp(in_features=self.in_channles, out_features=1)
        self.Res = Resdual_DoubleConv(nums=self.res_nums, in_channels=1)
        # self.Mfcc = MFFC_1D(channels=1, hidden_kernels=self.hidden_kernels)
        if self.single_mffc:
            self.Mffc = MFFC_1D(channels=1, hidden_kernels=self.hidden_kernels)
        else:
            self.Mffc = CMFFC_1D(channels=1, hidden_kernels=self.hidden_kernels)
        self.Linear = nn.Sequential(Mlp(in_features=self.feature_dim, out_features=self.num_classes),
                                    #nn.Linear(self.feature_dim, self.num_classes),
                                    nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.Norm(x)
        x = self.transformer_encoder_f(x)
        x = x.permute(0, 2, 1)
        x = self.down(x)
        x = x.permute(0, 2, 1)
        # x = self.Conv(x)
        x = self.Res(x)
        x = self.Mffc(x)
        # x = x.view(x.shape[0], -1)
        x = self.Linear(x)
        return x

class classifier_v2(nn.Module):
    def __init__(self, class_num, layer_num, resolution):
        super(classifier_v2, self).__init__()
        self.class_num = class_num
        self.layer_num = layer_num
        self.resolution = resolution
        self.bulit_model()

    def forward(self, x):
        B = x.shape[0]
        x = self.blocks(x)
        x = x.contiguous().view(B, -1)
        x = self.linear(x)
        x = self.softmax(x)
        return x

    def bulit_model(self):
        blocks = []
        for i in range(self.layer_num):
            blocks.append(Resdual_DoubleConv_2D())
        blocks.append(SFB_SC_2D(channels=1))
        self.blocks = nn.Sequential(*blocks)
        infs = 1
        for i in self.resolution:
            infs *= i
        self.linear = nn.Linear(in_features=infs, out_features=self.class_num)
        self.softmax = nn.Softmax(dim=1)



class Statue_1_Model(nn.Module):
    def __init__(self, feature_names, Spec_param, Wav_param, Fur_param, device):
        super(Statue_1_Model, self).__init__()
        self.feature_names = feature_names
        self.Spec_param = Spec_param
        self.Wav_param = Wav_param
        self.Fur_param = Fur_param
        self.DEVICE = device
        self.bulit_model()

    def bulit_model(self):
        feature_modules = {}
        fur_encoder_modules = {}
        for feature in self.feature_names:
            assert feature in WAVE_FEATURES or feature in SPEC_FEATURES, "wrong feature names @ " + feature
            if feature in WAVE_FEATURES:
                feature_modules[feature] = wave_encoder(res_nums=self.Wav_param["res_nums"],
                                                        channels=self.Wav_param["channels"],
                                                        hidden_kernels=self.Wav_param["hidden_kernels"],
                                                        resolution=self.Wav_param["resolution"],
                                                        hidden_features=self.Wav_param["hidden_features"],
                                                        single_mffc=self.Wav_param["single_mffc"])
            else:
                feature_modules[feature] = spec_encoder(res_nums=self.Spec_param["res_nums"],
                                                        channels=self.Spec_param["channels"],
                                                        hidden_kernels=self.Spec_param["hidden_kernels"],
                                                        resolution=self.Spec_param["resolution"],
                                                        hidden_features=self.Spec_param["hidden_features"],
                                                        single_mffc=self.Wav_param["single_mffc"])

            fur_encoder_modules[feature] = fur_encoder(res_nums=self.Fur_param["res_nums"],
                                                       in_features=self.Fur_param["in_features"],
                                                       out_features=self.Fur_param["out_features"])
        self.feature_modules = nn.ModuleDict(feature_modules)
        self.fur_encoder_modules = nn.ModuleDict(fur_encoder_modules)

    def forward(self, X):
        h_s, z_s = {}, {}
        for feature in self.feature_names:
            h_s[feature] = [self.feature_modules[feature](X[feature][0].to(self.DEVICE)),
                            self.feature_modules[feature](X[feature][1].to(self.DEVICE))]
            z_s[feature] = [self.fur_encoder_modules[feature](h_s[feature][0].to(self.DEVICE)),
                            self.fur_encoder_modules[feature](h_s[feature][1].to(self.DEVICE))]
        return h_s, z_s


if __name__ == '__main__':
    ins_spec = torch.rand(5, 128, 138)
    ins_wave = torch.rand(5, 4410)
    # model = CMFFC_1D(channels=1, hidden_kernels=(3, 5, 7, 9))
    # r = model(ins_wave)
    # model = wave_encoder(channels=1, hidden_kernels=(3, 5, 7, 9, 11))
    # r = model(ins_wave)
    # sub_model = fur_encoder(res_nums=5, in_features=256, out_features=128)
    # r_sub = sub_model(r)
    # r_sub = r_sub.unsqueeze(1)
    # fea_concat = torch.cat((r_sub, r_sub), dim=1)
    # cler = classifier(in_channles=2, feature_dim=128, encoder_num=5)
    # r = cler(fea_concat)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Full model Test Stage_1
    feature_names = ("wave", "fft", "mel")#, "stft", "temp")
    from work_space.config import Spec_param, Wav_param, Fur_param, Cls_param_v2, Cls_param
    test_model = Statue_1_Model(feature_names=feature_names,
                                Spec_param=Spec_param,
                                Wav_param=Wav_param,
                                Fur_param=Fur_param,
                                device=device).to(device)
    X = {"wave": [torch.rand(16, 4410), torch.rand(16, 4410)],
         "fft": [torch.rand(16, 4410), torch.rand(16, 4410)],
         "mel": [torch.rand(16, 128, 89), torch.rand(16, 128, 89)],
         "stft": [torch.rand(16, 128, 89), torch.rand(16, 128, 89)],
         "temp": [torch.rand(16, 128, 89), torch.rand(16, 128, 89)]}
    h_s, z_s = test_model(X)

    classifier_model = classifier_v1(in_channles=len(feature_names),
                                  feature_dim=Cls_param["feature_dim"],
                                  encoder_num=Cls_param["encoder_num"],
                                  hidden_kernels=Cls_param["hidden_kernels"],
                                  num_classes=Cls_param["num_classes"],
                                  res_nums=Cls_param["res_nums"],
                                  single_mffc=Cls_param["single_mffc"]).to(device)
    # classifier_model = classifier_v2(class_num=Cls_param_v2["class_num"],
    #                                  layer_num=Cls_param_v2["layer_num"],
    #                                  resolution=Cls_param_v2["resolution"]).to(device)

    joint_x = []
    for feature in feature_names:
        joint_x.append(z_s[feature][0].unsqueeze(1))
    x = torch.concat(joint_x, dim=1)
    # x = torch.randn_like(x)
    pred = classifier_model(x)#.unsqueeze(1))
    ans = torch.argmax(pred.squeeze(1), dim=1)
    ans = ans.cpu().detach().numpy()
