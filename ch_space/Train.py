import os
import torch
import shutil
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data.dataloader import DataLoader

from DataProcess.DataSet.GIS import GIS_sound_v3
from modules.encoder import Statue_1_Model
from modules.loss import *
from work_space.config import DataSet_param, Spec_param, Wav_param, Fur_param, feature_names, Cls_param, Crit_param

save_base_path = r""
Dataset_path = r""

num_epochs = DataSet_param["num_epochs"]
batch_size = DataSet_param["batch_size"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INDEX = 1

GIS_DataSet_Train = GIS_sound_v3(statue="train", file_train="train.csv",
                                 jitter_ratio=DataSet_param["jitter_ratio"],
                                 jitter_scale_ratio=DataSet_param["jitter_scale_ratio"],
                                 max_seg=DataSet_param["max_seg"],
                                 max_f_num=DataSet_param["max_f_num"],
                                 max_t_num=DataSet_param["max_t_num"],
                                 freq_mask_rate=DataSet_param["freq_mask_rate"],
                                 time_mask_rate=DataSet_param["time_mask_rate"],
                                 random_move=DataSet_param["random_move"],
                                 max_move_num=DataSet_param["max_move_num"])
GIS_DataLoader_Train = DataLoader(GIS_DataSet_Train, batch_size=batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, prefetch_factor=20)

GIS_DataSet_eva = GIS_sound_v3(statue="test", file_train="train.csv",
                               jitter_ratio=DataSet_param["jitter_ratio"],
                               jitter_scale_ratio=DataSet_param["jitter_scale_ratio"],
                               max_seg=DataSet_param["max_seg"],
                               max_f_num=DataSet_param["max_f_num"],
                               max_t_num=DataSet_param["max_t_num"],
                               freq_mask_rate=DataSet_param["freq_mask_rate"],
                               time_mask_rate=DataSet_param["time_mask_rate"],
                               random_move=DataSet_param["random_move"],
                               max_move_num=DataSet_param["max_move_num"])
GIS_DataLoader_eva = DataLoader(GIS_DataSet_eva, batch_size=batch_size, shuffle=True, num_workers=0,#2,
                                  pin_memory=True)#, prefetch_factor=20)


def create_set(Set):
    Sets = []
    for i in range(2, len(Set) + 1):     # 2 features can make groups
        iter = itertools.combinations(Set, i)
        iter = list(iter)
        for sub_set in iter:
            Sets.append(sub_set)
    return Sets


def calcuate_loss(h_s, z_s, features, batch_numbers, device, Crit_param):
    feature_arrangements = list(itertools.combinations(features, 2))
    nt_xent_criterion = NTXentLoss_poly(device=device, batch_size=batch_numbers,
                                        temperature=Crit_param["temperature"],
                                        use_cosine_similarity=Crit_param["use_cosine_similarity"]).to(device)
    loss = torch.tensor(0., device=device)
    loss_p_a = torch.tensor(0., device=device)
    loss_p_b = torch.tensor(0., device=device)
    loss_dict = {}
    for feature_1, feature_2 in feature_arrangements:
        # h - loss
        loss_a = nt_xent_criterion(h_s[feature_1][0], h_s[feature_1][1])
        loss_b = nt_xent_criterion(h_s[feature_2][0], h_s[feature_2][1])

        # z - loss
        l_ab = nt_xent_criterion(z_s[feature_1][0], z_s[feature_2][0])

        l_1, l_2, l_3 = nt_xent_criterion(z_s[feature_1][0], z_s[feature_2][1]), \
                        nt_xent_criterion(z_s[feature_1][1], z_s[feature_2][0]), \
                        nt_xent_criterion(z_s[feature_1][1], z_s[feature_2][1])

        loss_c = torch.pow(torch.pow(l_ab - l_1, 2) - Crit_param["delta"], 2) + \
                 torch.pow(torch.pow(l_ab - l_2, 2) - Crit_param["delta"], 2)

        loss = Crit_param["lambda_a"] * (loss_a + loss_b) + \
               Crit_param["lambda_b"] * loss_c + \
               Crit_param["lambda_c"] * l_ab + loss
        loss_p_a = (loss_a + loss_b) + loss_p_a
        loss_p_b = l_ab + loss_p_b
        loss_dict[feature_1 + "_" + feature_2] = [(loss_a + loss_b).item(), loss_c.item(), l_ab.item()]
    loss = -loss / (len(feature_arrangements) * INDEX)

    return loss, loss_dict, loss_p_a, loss_p_b


def train(dataloader, model, features, device, Crit_param, optimizer):
    model.train()
    model.to(device)
    optimizer.zero_grad()
    full_loss_dict = {}
    loss_s = []
    for data in dataloader:
        label = data["label"]
        batch_numbers = label.shape[0]
        h_s, z_s = model(data)
        loss, loss_dict, loss_p_a, loss_p_b = calcuate_loss(h_s=h_s, z_s=z_s, features=features,
                                        batch_numbers=batch_numbers, device=device,
                                        Crit_param=Crit_param)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_s.append(loss.item())
        for id_ in loss_dict:
            if id_ not in full_loss_dict.keys():
                full_loss_dict[id_] = [loss_dict[id_]]
            else:
                full_loss_dict[id_].append(loss_dict[id_])
    return loss_s, full_loss_dict


def test(dataloader, model, features, device, Crit_param):
    model.to(device)
    full_loss_dict = {}
    loss_s = []
    for data in dataloader:
        label = data["label"]
        batch_numbers = label.shape[0]
        with torch.no_grad():
            h_s, z_s = model(data)
            loss, loss_dict, loss_p_a, loss_p_b = calcuate_loss(h_s=h_s, z_s=z_s, features=features,
                                            batch_numbers=batch_numbers, device=device,
                                            Crit_param=Crit_param)
            loss_s.append(loss.item())
            for id_ in loss_dict:
                if id_ not in full_loss_dict.keys():
                    full_loss_dict[id_] = [loss_dict[id_]]
                else:
                    full_loss_dict[id_].append(loss_dict[id_])
    return loss_s, full_loss_dict


if __name__ == '__main__':
    train_sets = create_set(feature_names)
    trained = []
    for f_ in os.listdir(save_base_path):
        csvf = pd.read_csv(os.path.join(save_base_path, f_, "features_.csv")).values[:, 1]
        trained.append(tuple(csvf))

    for i, features_ in enumerate(train_sets):
        if features_ in trained:
            continue
        save_list_train, save_list_test = [], []
        full_save_dict_train, full_save_dict_test = {}, {}
        save_folder = os.path.join(save_base_path, "CASE_" + str(i+1))
        model = Statue_1_Model(feature_names=features_, Spec_param=Spec_param,
                               Wav_param=Wav_param, Fur_param=Fur_param, device=device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        pd.DataFrame(features_).to_csv(os.path.join(save_folder, "features_.csv"))
        torch.save(model.state_dict(), os.path.join(save_folder, "model.pth"))
        for epoch in tqdm(range(num_epochs)):
            loss_s_train, full_loss_dict_train = train(dataloader=GIS_DataLoader_Train,
                                                      model=model, features=features_,
                                                      device=device, Crit_param=Crit_param,
                                                      optimizer=optimizer)
            save_list_train.append(np.mean(loss_s_train))

            for id_ in full_loss_dict_train:
                if id_ not in full_save_dict_train.keys():
                    full_save_dict_train[id_] = [full_loss_dict_train[id_]]
                else:
                    full_save_dict_train[id_].append(full_loss_dict_train[id_])





