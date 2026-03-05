import torch
from hparams import hparams
from dataset import feature_stft, feature_contex
import os
import soundfile as sf
import numpy as np
import librosa

def eval_file_BN(wav_file, model, para):
    # 读取noisy 的音频文件
    noisy_wav, fs = sf.read(wav_file, dtype='int16')
    noisy_wav = noisy_wav.astype('float32')

    # 提取LPS特征
    noisy_LPS, noisy_phase = feature_stft(noisy_wav, para.para_stft)

    # 转为torch格式
    noisy_LPS = torch.from_numpy(noisy_LPS)

    # 进行拼帧
    noisy_LPS_expand = feature_contex(noisy_LPS, para.n_expand)

    # 利用DNN进行增强
    model.eval()
    with torch.no_grad():
        enh_LPS = model(x=noisy_LPS_expand, istraining=False)

    # 利用 BN-layer的信息对数据进行还原
    model_dic = model.state_dict()
    BN_weight = model_dic['BNlayer.weight'].data
    BN_weight = torch.unsqueeze(BN_weight, dim=0)

    BN_bias = model_dic['BNlayer.bias'].data
    BN_bias = torch.unsqueeze(BN_bias, dim=0)

    BN_mean = model_dic['BNlayer.running_mean'].data
    BN_mean = torch.unsqueeze(BN_mean, dim=0)

    BN_var = model_dic['BNlayer.running_var'].data
    BN_var = torch.unsqueeze(BN_var, dim=0)

    pred_LPS = (enh_LPS - BN_bias) * torch.sqrt(BN_var + 1e-4) / (BN_weight + 1e-8) + BN_mean

    # 将 LPS 还原成 Spec
    pred_LPS = pred_LPS.numpy()
    enh_mag = np.exp(pred_LPS.T / 2)
    enh_pahse = noisy_phase[para.n_expand:-para.n_expand, :].T
    enh_spec = enh_mag * np.exp(1j * enh_pahse)

    # istft
    enh_wav = librosa.istft(enh_spec, hop_length=para.para_stft["hop_length"], win_length=para.para_stft["win_length"])
    return enh_wav


if __name__ == "__main__":

    para = hparams()

    # 读取训练好的模型
    model_name = "save/model_15_0.00017.pth"
    # 注意: 如果模型是在 GPU 上训练的，但在 CPU 上加载，需要 map_location
    m_model = torch.load(model_name, map_location=torch.device('cpu'))

    # 定义文件路径和输出目录
    scp_file = 'scp/eval.scp'
    # 假设带噪语音文件的完整路径需要拼接一个基础路径,如果 eval.scp 中已经是完整路径，则不需要 eval_noisy_path
    eval_noisy_path = '/data1/BOX/eval_noisy/babble/5/'  # 请根据您的实际路径修改

    path_eval = 'eval_enhanced/babble/5/'  # 增强后语音的保存目录
    os.makedirs(path_eval, exist_ok=True)  # 创建输出目录

    # 读取带噪语音文件列表
    try:
        # 假设 eval.scp 包含的是文件名，每行一个
        noisy_files_list = np.loadtxt(scp_file, dtype='str').tolist()
    except Exception as e:
        print(f"Error reading {scp_file}: {e}")
        noisy_files_list = []

    print(f"Files to process: {len(noisy_files_list)}")

    # 遍历文件列表并进行增强
    for noisy_filename in noisy_files_list:

        noisy_file_full_path = os.path.join(eval_noisy_path, noisy_filename)

        # 构造增强后文件的保存路径
        # 例如: 'eval_enhanced/enh-Train_File1116.wav'
        file_id = os.path.split(noisy_filename)[-1]  # 提取文件名部分
        enh_filename = 'enh-' + file_id
        enh_file_path = os.path.join(path_eval, enh_filename)

        print(f"\nProcessing: {noisy_file_full_path}")

        # 进行增强
        try:
            enh_data = eval_file_BN(noisy_file_full_path, m_model, para)

            if enh_data.size == 0:
                print(f"Skipping {file_id}: Enhancement returned empty data.")
                continue

            max_ = np.max(enh_data)
            min_ = np.min(enh_data)
            # 这里的正则化是线性缩放到 [-1, 1]
            enh_data = enh_data * (2 / (max_ - min_)) - (max_ + min_) / (max_ - min_)

            _, fs = sf.read(noisy_file_full_path, dtype='int16')

            sf.write(enh_file_path, enh_data, fs)
            print(f"Saved enhanced file to: {enh_file_path}")

        except Exception as e:
            print(f"Error processing {noisy_file_full_path}: {e}")

    print("\nBatch enhancement complete.")
