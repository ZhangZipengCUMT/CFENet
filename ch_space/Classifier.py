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
from work_space.config import DataSet_param, Spec_param, Wav_param, Fur_param, feature_names, Cls_param, Crit_param


save_base_path = r""
Dataset_path = r""
seed = 42


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

def load_base_model(features, Spec_param, Wav_param, Fur_param, device, weight_path):
    model = Statue_1_Model(feature_names=features, Spec_param=Spec_param,
                           Wav_param=Wav_param, Fur_param=Fur_param, device=device)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    return model


def Classifiy(model, classifier, dataloader, optimizer, crit, features):
    model.to(device)
    classifier.to(device)
    crit.to(device)
    classifier.train()
    optimizer.zero_grad()
    acc_s, loss_cls = [], []
    for data in dataloader:
        label = data["label"]
        with torch.no_grad():
            h_s, z_s = model(data)
        joint_x = []
        for feature in features:
            joint_x.append(z_s[feature][0].unsqueeze(1))
        x = torch.concat(joint_x, dim=1)
        pred = classifier(x)#.unsqueeze(1))
        pred = pred.squeeze(1)
        loss = crit(pred, label.long().squeeze(1).to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_cls.append(loss.item())
        ans = torch.argmax(pred, dim=1)
        ans = ans.cpu().detach().numpy()
        label = label.squeeze(1).detach().numpy()
        for A, L in zip(ans, label):
            acc_s.append([A, L])
    acc_s = np.array(acc_s)
    return sum(acc_s[:, 0] == acc_s[:, 1]) / len(acc_s), np.mean(loss_cls), Counter(acc_s[:, 0])




if __name__ == '__main__':
    for folder in os.listdir(save_base_path):
        if os.path.exists(os.path.join(save_base_path, folder, "class_acc.csv")):
            continue
        weight_path = os.path.join(save_base_path, folder, "model.pth")
        features = pd.read_csv(os.path.join(save_base_path, folder, "features_.csv")).values[:, 1]
        model = load_base_model(tuple(features), Spec_param=Spec_param,
                                Wav_param=Wav_param, Fur_param=Fur_param,
                                device=device, weight_path=weight_path)
        classifier_model = classifier_v1(in_channles=len(features),
                                      feature_dim=Cls_param["feature_dim"],
                                      encoder_num=Cls_param["encoder_num"],
                                      hidden_kernels=Cls_param["hidden_kernels"],
                                      num_classes=Cls_param["num_classes"],
                                      res_nums=Cls_param["res_nums"])
        optimizer = optim.Adam(classifier_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        cert_cls = torch.nn.CrossEntropyLoss().to(device=device)
        acc_s, loss_s = [], []
        _tqdm = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            acc_, loss_, c = Classifiy(model=model, classifier=classifier_model,
                                    dataloader=GIS_DataLoader_Train, optimizer=optimizer,
                                    crit=cert_cls, features=features)
            acc_s.append(acc_)
            loss_s.append(loss_)
            _tqdm.update(1)
            _tqdm.set_postfix(acc='{:.6f}'.format(acc_))
            print(c)
        torch.save(classifier_model.state_dict(), os.path.join(save_base_path, folder, "classifier.pth"))
        pd.DataFrame(acc_s).to_csv(os.path.join(save_base_path, folder, "class_acc.csv"))
        pd.DataFrame(loss_s).to_csv(os.path.join(save_base_path, folder, "class_loss.csv"))

