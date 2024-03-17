from torch import nn
DataSet_param = {"jitter_ratio": 0.001,
                 "jitter_scale_ratio": 0.001,
                 "max_seg": 2,
                 "max_f_num": 2,
                 "max_t_num": 3,
                 "freq_mask_rate": 0.05,
                 "time_mask_rate": 0.05,
                 "random_move": True,
                 "max_move_num": 3,
                 "num_epochs": 150,
                 "num_epochs_pre": 15,
                 "batch_size": 32}

Crit_param = {"temperature": 0.2,
              "use_cosine_similarity": True,
              "lambda_a": 0.4,
              "lambda_b": 0.4,
              "lambda_c": 0.2,
              "delta": 0.5,
              "lambda_class_p1": 0.99,
              "lambda_class_p2": -0.005,
              "lambda_class_p3": -0.005}

Spec_param = {"res_nums": 1,
              "channels": 1,
              "hidden_kernels": (3,),
              "resolution": (128, 89),
              "hidden_features": 512,
              "single_mffc": False}

Wav_param = {"res_nums": 1,
             "channels": 1,
             "hidden_kernels": (3,),
             "resolution": (4410,),
             "hidden_features": 512,
             "single_mffc": False}

Fur_param = {"res_nums": 0,
             "in_features": 512,
             "out_features": 256}

feature_names = ("wave", "fft", "stft", "mel")#, "temp")

Cls_param = {"in_channles": 2,  # todo
             "feature_dim": 256,
             "encoder_num": 1,
             "hidden_kernels": (3,),
             "res_nums": 1,
             "num_classes": 4,
             "single_mffc": False}

Cls_param_v2 = {"class_num": 4,
                "layer_num": 5,
                "resolution": (3, 256)}