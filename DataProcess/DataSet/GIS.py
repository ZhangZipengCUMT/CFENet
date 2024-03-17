import numpy as np
import os
import librosa
import pandas as pd
import torch
import torch.nn.functional as F
from feature.HPCP import hpcp
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
# from torchaudio import
import copy
from sklearn import preprocessing
import random
from DataProcess.DataArgument.augmentations import DataTransform_TD_bank, DataTransform_FD, spec_augment

class GIS(Dataset):
    def __init__(self,Datapath = r"D:\DataBase\GIS",statue= "train",num = 5):
        super(GIS, self).__init__()
        folder = os.listdir(Datapath)
        self.statue = statue
        # create list with different use
        if statue=="train":
            self.train_list = []
        elif statue=="full":
            self.file_list = []
        else:
            self.val_list = []
        self.onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        # init encoder
        classname = []
        for c in folder:
            folder_path = os.path.join(Datapath,c)
            if os.path.isdir(folder_path): # make judgement on folder name
                classname.append(c)
        classname = np.array(classname).reshape((-1,1))
        self.onehot_encoder.fit(classname)
        print("encoder init done")
        # init file list
        count = 0
        for f in folder:
            folder_path = os.path.join(Datapath,f)
            if os.path.isdir(folder_path):              # make judgement on folder name
                wav_list = os.listdir(folder_path)
                for wav in wav_list:
                    wav_path, label = os.path.join(folder_path, wav), f
                    if statue == "full":
                        self.file_list.append([wav_path, label])
                    elif statue=="train":
                        if count%num!=0:
                            self.train_list.append([wav_path,label])
                    else:
                        if count%num == 0:
                            self.val_list.append([wav_path,label])
                    count+=1
        print("file list init done")

    def __getitem__(self, item):
        if self.statue=="train":
            wav_path, label = self.train_list[item]
        elif self.statue=="full":
            wav_path, label = self.file_list[item]
        else:
            wav_path, label = self.val_list[item]
        wav, sr = librosa.load(wav_path)
        # S = librosa.feature.melspectrogram(y=y, sr=sr)
        # S_dB = librosa.power_to_db(S, ref=np.max)
        # wav = S_dB
        label = self.onehot_encoder.transform(np.array(label).reshape((-1,1)))#.indptr # remember real TM grave
        return wav,label
    def __len__(self):
        if self.statue=="train":
            return len(self.train_list)
        elif self.statue=="full":
            return len(self.file_list)
        else:
            return len(self.val_list)

class GIS_sound(Dataset): # sound DataSet
    def __init__(self, Datapath=r"D:\DataBase\GIS_0.1\split", statue="train",
                 percent=0.7, require_mel=False, onehot=True, show_getting=False):
        random.seed(42) # legend 23
        np.random.seed(42)
        super(GIS_sound, self).__init__()
        self.statue = statue
        self.require_mel = require_mel
        self.show_getting = show_getting
        # self.Task_2_transform = transforms.Compose([transforms.ToTensor(),
        #                                             transforms.Normalize(0.5, 0.5)])
        # create list with different use
        if statue == "train":
            self.train_list = []
        elif statue == "val":
            self.val_list = []
        else:
            self.file_list = []

        # init one hot encoder
        if onehot:
            self.onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
            classes = np.array([c for c in os.listdir(Datapath)]).reshape((-1,1))
            self.onehot_encoder.fit(classes)
        else:
            self.LabelEncoder = preprocessing.LabelEncoder()
            classes = np.array([c for c in os.listdir(Datapath)]).reshape((-1, 1))
            self.LabelEncoder.fit(classes)

        # one hot for frequency
        self.onehot_encoder_f = preprocessing.OneHotEncoder(sparse=False)
        classes_f = []

        for fault in os.listdir(Datapath):
            fault_path = os.path.join(Datapath,fault)

            for frequency in os.listdir(fault_path):
                if frequency not in classes_f:
                    classes_f.append(frequency)
                fault_frequency_path = os.path.join(fault_path,frequency)
                full_list = np.array([os.path.join(fault_frequency_path,p) for p in os.listdir(fault_frequency_path)])
                choose_list = np.random.choice(full_list,int(len(full_list) * percent),replace = False)
                rest_list = full_list[~np.isin(full_list,choose_list)]
                if self.statue == "train":
                    for file in choose_list:
                        self.train_list.append([fault, int(frequency), file])
                elif self.statue == "val":
                    for file in rest_list:
                        self.val_list.append([fault, int(frequency), file])
                else: # full
                    for file in full_list:
                        self.file_list.append([fault,int(frequency),file])
        print("file list init done")
        self.onehot = onehot
        classes_f = np.array(classes_f).reshape(-1, 1)
        self.onehot_encoder_f.fit(classes_f)

    def __getitem__(self, item):
        if self.show_getting:
            print("Reading ", item, " into memory")
        if self.statue == "train":
            fault, frequency, path = self.train_list[item]
        elif self.statue == "val":
            fault, frequency, path = self.val_list[item]
        else:
            fault, frequency, path = self.file_list[item]
        wave_data, sr = librosa.load(path, sr=44100)
        if self.onehot:
            label = self.onehot_encoder.transform(np.array(fault).reshape((-1, 1)))
        else:
            label = self.LabelEncoder.transform(np.array(fault).reshape((-1, 1)))
        if self.require_mel:
            S = librosa.feature.melspectrogram(y=wave_data, sr=sr, n_fft=1024, hop_length=32)#37)#32)
            S_dB = librosa.power_to_db(S, ref=np.max)
            # wave_data = self.Task_2_transform(wave_data)
            # S_dB = self.Task_2_transform(S_dB)
            S_dB = 2 * (S_dB - S_dB.min())/(S_dB.max() - S_dB.min()) - 1
            return label, frequency/180, wave_data, sr, S_dB
        # wave_data = wave_data[0]
        else:
            # frequency = self.onehot_encoder_f.transform(np.array(str(frequency)).reshape((-1, 1)))
            # wave_data = self.Task_2_transform(wave_data)
            # S_dB = self.Task_2_transform.transforms(S_dB)
            return label, frequency/180, wave_data, sr

    def __len__(self):
        if self.statue == "train":
            return len(self.train_list)
        elif self.statue == "val":
            return len(self.val_list)
        else:
            return len(self.file_list)


class GIS_sound_v3(Dataset): # sound DataSet
    def __init__(self, Datapath=r"D:\ZZP_IDP_Base\Data\GIS_0.1", statue="train", file_train="train.csv", save_name="train.csv",
                 percent=0.7, datalen=None, jitter_ratio=0.001, jitter_scale_ratio=0.001, max_seg=5,
                 max_f_num=2, max_t_num=3, freq_mask_rate=0.3, time_mask_rate=0.05, random_move=True, max_move_num=3):
        random.seed(42)      # legend 23
        np.random.seed(42)
        super(GIS_sound_v3, self).__init__()
        self.statue = statue

        self.jitter_ratio = jitter_ratio
        self.jitter_scale_ratio = jitter_scale_ratio
        self.max_seg = max_seg
        self.max_f_num = max_f_num
        self.max_t_num = max_t_num
        self.freq_mask_rate = freq_mask_rate
        self.time_mask_rate = time_mask_rate
        self.random_move = random_move
        self.max_move_num = max_move_num



        self.base_path = Datapath
        self.datalen = datalen

        self.file_list = []
        if file_train is not None:
            try:
                data = pd.read_csv(file_train).values[:, 1:]
                if statue == "train":
                    self.file_list = data
            except:
                data = None
        else:
            data = None

        self.LabelEncoder = preprocessing.LabelEncoder()
        classes = np.array([c for c in os.listdir(Datapath)]).reshape((-1, 1))
        self.LabelEncoder.fit(classes)


        if len(self.file_list) == 0 or file_train is None or statue != "train":
            train_list = []
            for fault in os.listdir(Datapath):
                fault_path = os.path.join(Datapath, fault)
                for frequency in os.listdir(fault_path):
                    fault_frequency_path = os.path.join(fault_path, frequency)
                    full_list = np.array([os.path.join(fault, frequency, p) for p in os.listdir(fault_frequency_path)])
                    choose_list = np.random.choice(full_list, int(len(full_list) * percent), replace=False)
                    rest_list = full_list[~np.isin(full_list, choose_list)]

                    if self.statue == "train":
                        for file in choose_list:
                            self.file_list.append([fault, int(frequency), file])
                            train_list.append([fault, int(frequency), file])
                    else:
                        for file in full_list:
                            # c = data[:, 2]
                            if data is None or file not in data[:, 2]:
                                self.file_list.append([fault, int(frequency), file])
                            # else:
                            #     print(66)
            if file_train is None:
                pd.DataFrame(train_list).to_csv(save_name)

    def __getitem__(self, item):
        fault, frequency, path = self.file_list[item]
        wave_data, sr = librosa.load(os.path.join(self.base_path, path), sr=44100)



        fft = abs(np.fft.fft(wave_data))
        fft = (fft - fft.min()) / (fft.max() - fft.min())
        fft_arg = DataTransform_FD(copy.deepcopy(fft))

        label = self.LabelEncoder.transform(np.array(fault).reshape((-1, 1)))

        S = librosa.feature.melspectrogram(y=wave_data, sr=sr, n_fft=512, hop_length=50, win_length=128)#37)#32)
        S_dB = librosa.power_to_db(S, ref=np.max)
        g = S_dB.min()
        melspectrogram = (S_dB - S_dB.min())/(S_dB.max() - S_dB.min())
        melspectrogram_arg = spec_augment(copy.deepcopy(melspectrogram),
                                          max_f_num=self.max_f_num,
                                          max_t_num=self.max_t_num,
                                          freq_mask_rate=self.freq_mask_rate,
                                          time_mask_rate=self.time_mask_rate,
                                          random_move=self.random_move,
                                          max_move_num=self.max_move_num)


        stft = librosa.feature.chroma_stft(y=wave_data, sr=44100, n_fft=512, hop_length=50, n_chroma=128)
        stft = (stft - stft.min())/(stft.max() - stft.min())
        stft_arg = spec_augment(copy.deepcopy(stft),
                                max_f_num=self.max_f_num,
                                          max_t_num=self.max_t_num,
                                          freq_mask_rate=self.freq_mask_rate,
                                          time_mask_rate=self.time_mask_rate,
                                          random_move=self.random_move,
                                          max_move_num=self.max_move_num)

        tempogram = librosa.feature.tempogram(y=wave_data, sr=44100, hop_length=50, win_length=128)
        tempogram = (tempogram - tempogram.min()) / (tempogram.max() - tempogram.min())
        tempogram_arg = spec_augment(copy.deepcopy(tempogram), max_f_num=self.max_f_num,
                                          max_t_num=self.max_t_num,
                                          freq_mask_rate=self.freq_mask_rate,
                                          time_mask_rate=self.time_mask_rate,
                                          random_move=self.random_move,
                                          max_move_num=self.max_move_num)


        wave_data = (wave_data - wave_data.min()) / (wave_data.max() - wave_data.min())
        wave_arg = DataTransform_TD_bank(copy.deepcopy(wave_data), jitter_ratio=self.jitter_ratio,
                                         jitter_scale_ratio=self.jitter_scale_ratio, max_seg=self.max_seg)
        if self.datalen is not None:
            wave_data = wave_data[:self.datalen]

        return {"label": label, "wave": (wave_data.astype(np.float32), wave_arg.astype(np.float32)),
                "fft": (fft.astype(np.float32), fft_arg.astype(np.float32)),
                "mel": (melspectrogram.astype(np.float32), melspectrogram_arg.astype(np.float32)),
                "stft": (stft.astype(np.float32), stft_arg.astype(np.float32)),
                "temp": (tempogram.astype(np.float32), tempogram_arg.astype(np.float32))}

    def __len__(self):
        return len(self.file_list)



class GIS_sound_v4(Dataset): # sound DataSet
    def __init__(self, Datapath=r"D:\ZZP_IDP_Base\Data\GIS_0.1", statue="train", file_train="train.csv", save_name="train.csv",
                 percent=0.7, datalen=None, jitter_ratio=0.001, jitter_scale_ratio=0.001, max_seg=5,
                 max_f_num=2, max_t_num=3, freq_mask_rate=0.3, time_mask_rate=0.05, random_move=True, max_move_num=3,
                 SNR=0):
        random.seed(42)      # legend 23
        np.random.seed(42)
        super(GIS_sound_v4, self).__init__()
        self.statue = statue

        self.jitter_ratio = jitter_ratio
        self.jitter_scale_ratio = jitter_scale_ratio
        self.max_seg = max_seg
        self.max_f_num = max_f_num
        self.max_t_num = max_t_num
        self.freq_mask_rate = freq_mask_rate
        self.time_mask_rate = time_mask_rate
        self.random_move = random_move
        self.max_move_num = max_move_num
        self.SNR = SNR

        self.base_path = Datapath
        self.datalen = datalen

        self.file_list = []
        if file_train is not None:
            try:
                data = pd.read_csv(file_train).values[:, 1:]
                if statue == "train":
                    self.file_list = data
            except:
                data = None
        else:
            data = None

        self.LabelEncoder = preprocessing.LabelEncoder()
        classes = np.array([c for c in os.listdir(Datapath)]).reshape((-1, 1))
        self.LabelEncoder.fit(classes)


        if len(self.file_list) == 0 or file_train is None or statue != "train":
            train_list = []
            for fault in os.listdir(Datapath):
                fault_path = os.path.join(Datapath, fault)
                for frequency in os.listdir(fault_path):
                    fault_frequency_path = os.path.join(fault_path, frequency)
                    full_list = np.array([os.path.join(fault, frequency, p) for p in os.listdir(fault_frequency_path)])
                    choose_list = np.random.choice(full_list, int(len(full_list) * percent), replace=False)
                    rest_list = full_list[~np.isin(full_list, choose_list)]

                    if self.statue == "train":
                        for file in choose_list:
                            self.file_list.append([fault, int(frequency), file])
                            train_list.append([fault, int(frequency), file])
                    else:
                        for file in full_list:
                            # c = data[:, 2]
                            if data is None or file not in data[:, 2]:
                                self.file_list.append([fault, int(frequency), file])
                            # else:
                            #     print(66)
            if file_train is None:
                pd.DataFrame(train_list).to_csv(save_name)


    def add_noise(self, wave):
        noise = np.random.normal(0, 0.01, wave.shape)
        clean_p = np.sum(np.power(wave, 2))
        noise_p = np.sum(np.power(noise, 2))
        snr = self.SNR  #random.uniform(self.snr[0], self.snr[1])
        k = np.sqrt(clean_p / (noise_p * 10 ** (snr / 10)))
        #
        if np.isinf(k).any() or np.isnan(k).any():
            k = 0
        wave_data = wave + k * noise
        wave_data = (wave_data - wave_data.min()) / (wave_data.max() - wave_data.min())#todo
        return wave_data


    def __getitem__(self, item):
        fault, frequency, path = self.file_list[item]
        wave_data, sr = librosa.load(os.path.join(self.base_path, path), sr=44100)

        wave_data = self.add_noise(wave_data)

        fft = abs(np.fft.fft(wave_data))
        fft = (fft - fft.min()) / (fft.max() - fft.min())
        fft_arg = DataTransform_FD(copy.deepcopy(fft))

        label = self.LabelEncoder.transform(np.array(fault).reshape((-1, 1)))

        S = librosa.feature.melspectrogram(y=wave_data, sr=sr, n_fft=512, hop_length=50, win_length=128)#37)#32)
        S_dB = librosa.power_to_db(S, ref=np.max)
        g = S_dB.min()
        melspectrogram = (S_dB - S_dB.min())/(S_dB.max() - S_dB.min())
        melspectrogram_arg = spec_augment(copy.deepcopy(melspectrogram),
                                          max_f_num=self.max_f_num,
                                          max_t_num=self.max_t_num,
                                          freq_mask_rate=self.freq_mask_rate,
                                          time_mask_rate=self.time_mask_rate,
                                          random_move=self.random_move,
                                          max_move_num=self.max_move_num)


        stft = librosa.feature.chroma_stft(y=wave_data, sr=44100, n_fft=512, hop_length=50, n_chroma=128)
        stft = (stft - stft.min())/(stft.max() - stft.min())
        stft_arg = spec_augment(copy.deepcopy(stft),
                                max_f_num=self.max_f_num,
                                          max_t_num=self.max_t_num,
                                          freq_mask_rate=self.freq_mask_rate,
                                          time_mask_rate=self.time_mask_rate,
                                          random_move=self.random_move,
                                          max_move_num=self.max_move_num)

        tempogram = librosa.feature.tempogram(y=wave_data, sr=44100, hop_length=50, win_length=128)
        tempogram = (tempogram - tempogram.min()) / (tempogram.max() - tempogram.min())
        tempogram_arg = spec_augment(copy.deepcopy(tempogram), max_f_num=self.max_f_num,
                                          max_t_num=self.max_t_num,
                                          freq_mask_rate=self.freq_mask_rate,
                                          time_mask_rate=self.time_mask_rate,
                                          random_move=self.random_move,
                                          max_move_num=self.max_move_num)


        wave_data = (wave_data - wave_data.min()) / (wave_data.max() - wave_data.min())
        wave_arg = DataTransform_TD_bank(copy.deepcopy(wave_data), jitter_ratio=self.jitter_ratio,
                                         jitter_scale_ratio=self.jitter_scale_ratio, max_seg=self.max_seg)
        if self.datalen is not None:
            wave_data = wave_data[:self.datalen]

        return {"label": label, "wave": (wave_data.astype(np.float32), wave_arg.astype(np.float32)),
                "fft": (fft.astype(np.float32), fft_arg.astype(np.float32)),
                "mel": (melspectrogram.astype(np.float32), melspectrogram_arg.astype(np.float32)),
                "stft": (stft.astype(np.float32), stft_arg.astype(np.float32)),
                "temp": (tempogram.astype(np.float32), tempogram_arg.astype(np.float32))}

    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    # a = np.array(['haha','jfed']).reshape((-1,1))
    # onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    # onehot_encoder.fit_transform(a)
    # a = onehot_encoder.transform(np.array(['jfed']).reshape((-1,1)))#.indptr
    import tqdm
    da = GIS_sound_v4(statue="train")
    Dl = DataLoader(da, shuffle=True)
    a = []
    for data in tqdm.tqdm(Dl):
        label = data["label"]
        wave = data["wave"]
        fft = data["fft"]
        mel = data["mel"]
        stft = data["stft"]
        temp = data["temp"]


        # print(o[0].shape, o[1].shape, o[2].shape, o[3].shape, o[4].shape)