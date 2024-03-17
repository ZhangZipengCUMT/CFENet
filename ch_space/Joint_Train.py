import os
import torch
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data.dataloader import DataLoader

from modules.encoder import Statue_1_Model, classifier_v1, classifier_v2
from DataProcess.DataSet.GIS import GIS_sound_v3
from work_space.config import DataSet_param

from Train import train, calcuate_loss
from Classifier import Classifiy


save_base_path = r""
Dataset_path = r""
seed = 42

num_epochs_pre = DataSet_param["num_epochs_pre"]
num_epochs = DataSet_param["num_epochs"]
batch_size = DataSet_param["batch_size"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
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



def Joint_Classifiy(model, classifier, dataloader, Crit_param, optimizer, optimizer_c, crit, features):
    model.to(device)
    classifier.to(device)
    crit.to(device)
    classifier.train()
    optimizer.zero_grad()
    acc_s, loss_cls = [], []
    full_loss_dict = {}
    loss_s = []
    for data in dataloader:
        label = data["label"]
        batch_numbers = label.shape[0]
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
        loss_s.append(loss.item())

        joint_x = []
        for feature in features:
            joint_x.append(z_s[feature][0].unsqueeze(1))
        x = torch.concat(joint_x, dim=1)
        pred = classifier(x)
        pred = pred.squeeze(1)
        loss_c = crit(pred, label.long().squeeze(1).to(device))
        loss_f = Crit_param["lambda_class_p1"] * loss_c + \
                 Crit_param["lambda_class_p2"] * loss_p_a + \
                 Crit_param["lambda_class_p3"] * loss_p_b   # + loss
        loss_f.backward()

        loss_cls.append(loss_c.item())
        ans = torch.argmax(pred, dim=1)
        ans = ans.cpu().detach().numpy()
        label = label.squeeze(1).detach().numpy()
        for A, L in zip(ans, label):
            acc_s.append([A, L])

        optimizer.step()
        optimizer.zero_grad()
        optimizer_c.step()
        optimizer_c.zero_grad()


    acc_s = np.array(acc_s)
    return sum(acc_s[:, 0] == acc_s[:, 1]) / len(acc_s), np.mean(loss_cls), Counter(acc_s[:, 0]), loss_s, full_loss_dict

def Joint_Test(model, classifier, dataloader, features):
    model.to(device)
    classifier.to(device)
    classifier.train()
    acc_s, loss_cls = [], []
    full_loss_dict = {}
    loss_s = []
    for data in dataloader:
        label = data["label"]
        batch_numbers = label.shape[0]
        h_s, z_s = model(data)

        joint_x = []
        for feature in features:
            joint_x.append(z_s[feature][0].unsqueeze(1))
        x = torch.concat(joint_x, dim=1)
        pred = classifier(x)
        pred = pred.squeeze(1)

        ans = torch.argmax(pred, dim=1)
        ans = ans.cpu().detach().numpy()
        label = label.squeeze(1).detach().numpy()
        for A, L in zip(ans, label):
            acc_s.append([A, L])
    acc_s = np.array(acc_s)
    return sum(acc_s[:, 0] == acc_s[:, 1]) / len(acc_s)


def FULL_TRAIN_CLASS(GIS_DataLoader_Train, GIS_DataLoader_eva,
                     model, classifier_model,
                     feature_names, device, Crit_param, cert_cls,
                     optimizer, optimizer_c, save_base_path):
    acc_s, loss_s = [], []
    save_list_train, save_list_test = [], []
    full_save_dict_train, full_save_dict_test = {}, {}
    _tqdm_c = tqdm(total=num_epochs_pre)
    for epoch in range(num_epochs_pre):
        loss_s_train, full_loss_dict_train = train(dataloader=GIS_DataLoader_Train,
                                                   model=model, features=feature_names,
                                                   device=device, Crit_param=Crit_param,
                                                   optimizer=optimizer)
        _tqdm_c.update(1)
        _tqdm_c.set_postfix(loss='{:.6f}'.format(np.mean(loss_s_train)))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    torch.save(model.state_dict(), os.path.join(save_base_path, "model_pretrained.pth"))
    best_acc = 0
    _tqdm = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        acc_, loss_, c, loss_s_train, full_loss_dict_train = Joint_Classifiy(model=model, classifier=classifier_model,
                                                                             dataloader=GIS_DataLoader_Train,
                                                                             Crit_param=Crit_param, optimizer=optimizer,
                                                                             optimizer_c=optimizer_c, crit=cert_cls,
                                                                             features=feature_names)
        test_acc = Joint_Test(model=model, classifier=classifier_model,
                              dataloader=GIS_DataLoader_eva, features=feature_names)
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_base_path, "best_model.pth"))
            torch.save(classifier_model.state_dict(), os.path.join(save_base_path, "best_classifier.pth"))
        save_list_train.append(np.mean(loss_s_train))
        for id_ in full_loss_dict_train:
            if id_ not in full_save_dict_train.keys():
                full_save_dict_train[id_] = [full_loss_dict_train[id_]]
            else:
                full_save_dict_train[id_].append(full_loss_dict_train[id_])
        acc_s.append([acc_, test_acc])
        loss_s.append(loss_)
        _tqdm.update(1)
        _tqdm.set_postfix(acc='{:.6f}'.format(acc_))
        print(np.mean(loss_s_train), test_acc, loss_, c)

    pd.DataFrame(save_list_train).to_csv(os.path.join(save_base_path, "Train.csv"))
    pd.DataFrame(full_save_dict_train).to_csv(os.path.join(save_base_path, "Train_Detail.csv"))
    torch.save(model.state_dict(), os.path.join(save_base_path, "model.pth"))
    # shutil.copy("config.py", os.path.join(save_base_path, "config.py"))
    pd.DataFrame(feature_names).to_csv(os.path.join(save_base_path, "features_.csv"))

    torch.save(classifier_model.state_dict(), os.path.join(save_base_path, "classifier.pth"))
    pd.DataFrame(acc_s).to_csv(os.path.join(save_base_path, "class_acc.csv"))
    pd.DataFrame(loss_s).to_csv(os.path.join(save_base_path, "class_loss.csv"))

if __name__ == '__main__':
    features_ = ("fft", "mel")
    for i, abl_id in enumerate(["CH_1", "CH_2", "CH_3", "CH_4", "CH_5", "CH_6"]):
        save_folder = os.path.join(save_base_path, abl_id)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if abl_id == "CH_1":
            from config_ch_1 import Spec_param, Wav_param, Fur_param, Cls_param, Crit_param
            shutil.copy("config_ch_1.py", os.path.join(save_folder, "config.py"))
        elif abl_id == "CH_2":
            from config_ch_2 import Spec_param, Wav_param, Fur_param, Cls_param, Crit_param
            shutil.copy("config_ch_2.py", os.path.join(save_folder, "config.py"))
        elif abl_id == "CH_3":
            from config_ch_3 import Spec_param, Wav_param, Fur_param, Cls_param, Crit_param
            shutil.copy("config_ch_3.py", os.path.join(save_folder, "config.py"))
        elif abl_id == "CH_4":
            from config_ch_4 import Spec_param, Wav_param, Fur_param, Cls_param, Crit_param
            shutil.copy("config_ch_4.py", os.path.join(save_folder, "config.py"))
        elif abl_id == "CH_5":
            from config_ch_5 import Spec_param, Wav_param, Fur_param, Cls_param, Crit_param
            shutil.copy("config_ch_5.py", os.path.join(save_folder, "config.py"))
        elif abl_id == "CH_6":
            from config_ch_6 import Spec_param, Wav_param, Fur_param, Cls_param, Crit_param
            shutil.copy("config_ch_6.py", os.path.join(save_folder, "config.py"))


        model = Statue_1_Model(feature_names=features_, Spec_param=Spec_param,
                               Wav_param=Wav_param, Fur_param=Fur_param, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        classifier_model = classifier_v1(in_channles=len(features_),
                                         feature_dim=Cls_param["feature_dim"],
                                         encoder_num=Cls_param["encoder_num"],
                                         hidden_kernels=Cls_param["hidden_kernels"],
                                         num_classes=Cls_param["num_classes"],
                                         res_nums=Cls_param["res_nums"],
                                         single_mffc=Cls_param["single_mffc"]).to(device)
        optimizer_c = optim.Adam(classifier_model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        cert_cls = torch.nn.CrossEntropyLoss().to(device=device)


        pd.DataFrame(features_).to_csv(os.path.join(save_folder, "features_.csv"))
        torch.save(model.state_dict(), os.path.join(save_folder, "model.pth"))
        torch.save(classifier_model.state_dict(), os.path.join(save_folder, "classifier.pth"))

        FULL_TRAIN_CLASS(GIS_DataLoader_Train=GIS_DataLoader_Train,
                         GIS_DataLoader_eva=GIS_DataLoader_eva,
                         model=model,
                         classifier_model=classifier_model,
                         feature_names=features_,
                         device=device,
                         Crit_param=Crit_param,
                         cert_cls=cert_cls,
                         optimizer=optimizer,
                         optimizer_c=optimizer_c,
                         save_base_path=save_folder)
