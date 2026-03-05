import torch
from hparams import hparams
from dataset import feature_stft, feature_contex
import os
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from generate_training_data import signal_by_db


def eval_file_IRM(wav_file,model,para):
    
    # 读取noisy 的音频文件
    noisy_wav,fs = sf.read(wav_file,dtype = 'float32')
    noisy_wav = noisy_wav.astype('float32')

    # 提取LPS特征
    noisy_mag,noisy_phase = feature_stft(noisy_wav,para.para_stft)

    # 转为torch格式
    noisy_LPS = torch.from_numpy(np.log(noisy_mag**2))

    # 进行拼帧
    noisy_LPS_expand = feature_contex(noisy_LPS,para.n_expand)
    
    # 利用DNN进行mask计算
    model.eval()
    with torch.no_grad():
        enh_mask = model(x = noisy_LPS_expand)
    # 转为numpy格式
    enh_mask = enh_mask.numpy()

    enh_pahse = noisy_phase[para.n_expand:-para.n_expand,:].T
    enh_mag = (noisy_mag[para.n_expand:-para.n_expand,:]*enh_mask).T

    enh_spec = enh_mag*np.exp(1j*enh_pahse)

    # istft
    enh_wav = librosa.istft(enh_spec, hop_length=para.para_stft["hop_length"], win_length=para.para_stft["win_length"])
    return enh_wav


if __name__ == "__main__":

    para = hparams()
    device = torch.device('cpu')  # 如果有GPU可以改为 'cuda'

    # 1. 加载模型
    model_path = "save/model_59_0.0357.pth"
    # 加载模型并确保在 CPU 上（或指定设备）
    m_model = torch.load(model_path, map_location=device)
    m_model.eval()

    # 2. 设置路径
    # scp 文件里记录的是带噪音频的文件名（如：file1.wav）
    scp_file = 'scp/eval.scp'
    # 已有的带噪语音根目录
    eval_noisy_dir = '/data1/BOX/eval_noisy/babble/-5/'
    # 增强后的保存目录
    path_save = 'eval_enhanced/babble/-5/'
    os.makedirs(path_save, exist_ok=True)

    # 3. 读取待处理列表
    try:
        noisy_files = np.loadtxt(scp_file, dtype='str').tolist()
        if isinstance(noisy_files, str):  # 处理只有一个文件名的情况
            noisy_files = [noisy_files]
    except Exception as e:
        print(f"Read list error: {e}")
        noisy_files = []

    print(f"Processing {len(noisy_files)} files...")

    # 4. 循环处理
    for filename in noisy_files:
        # 拼接完整输入路径
        noisy_full_path = os.path.join(eval_noisy_dir, filename)

        # 检查文件是否存在
        if not os.path.exists(noisy_full_path):
            print(f"File not found: {noisy_full_path}")
            continue

        # 构造输出路径 (加上 'enh-' 前缀)
        file_id = os.path.basename(filename)
        save_path = os.path.join(path_save, 'enh-' + file_id)

        try:
            # 调用原本的 IRM 函数进行增强
            print(f"Enhancing: {file_id}")
            enh_data = eval_file_IRM(noisy_full_path, m_model, para)

            # 获取原始采样率用于保存
            _, fs = sf.read(noisy_full_path, frames=1)

            # --- 归一化处理 ---
            # 采用 Peak Normalization 防止写入时溢出导致杂音
            if np.max(np.abs(enh_data)) > 0:
                enh_data = enh_data / np.max(np.abs(enh_data))

            # 保存文件
            sf.write(save_path, enh_data, fs)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nBatch Enhancement Task Completed.")
                
                
                
                
               
    
 
    
    