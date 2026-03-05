"""
按照eval.scp文件中的语音数据进行噪声混合，并将混合好的噪声按不同信噪比保存到指定文件夹
"""
import os
import numpy as np
import soundfile as sf
from generate_training_data import signal_by_db
from hparams import hparams


def generate_evaluation_data(para):
    snrs = [-5, 0, 5]
    noise_path = '/data1/BOX/eval_noise/'
    noises = ['babble']

    scp_file = 'scp/eval.scp'
    test_clean_files = []

    try:
        with open(scp_file, 'r', encoding='utf-8') as f:
            test_clean_files = [line.strip() for line in f if line.strip()]

    except FileNotFoundError:
        print(f"错误：文件 '{scp_file}' 未找到！请检查路径。")
        return None, None, None, None

    # 限制为前 n 个文件,n 可修改
    test_clean_files = test_clean_files[:1680]

    path_eval_noisy = '/data1/BOX/eval_noisy/'
    clean_path = '/data1/BOX/eval_clean/'  # 纯净文件的根目录

    # 确保评估输出根目录存在
    os.makedirs(path_eval_noisy, exist_ok=True)
    print("--- 正在生成评估所需的带噪语音 ---")

    for noise in noises:
        print(f"处理噪声: {noise}")

        # 检查噪声文件是否存在
        noise_file = os.path.join(noise_path, noise + '.wav')
        if not os.path.exists(noise_file):
            print(f"错误: 噪声文件 {noise_file} 不存在!")
            continue

        noise_data, fs = sf.read(noise_file, dtype='int16')

        for clean_wav in test_clean_files:

            clean_file = os.path.join(clean_path, clean_wav)

            # 检查纯净语音文件是否存在
            if not os.path.exists(clean_file):
                print(f"错误: 纯净语音文件 {clean_file} 不存在! 跳过...")
                continue

            clean_data, fs = sf.read(clean_file, dtype='int16')
            id_ = os.path.split(clean_file)[-1]  # 提取文件名 ID

            for snr in snrs:
                # === 实现分层目录结构 ===
                # 目标目录: path_eval_noisy / noise / str(snr)
                output_dir = os.path.join(path_eval_noisy, noise, str(snr))
                os.makedirs(output_dir, exist_ok=True)  # 创建当前 snr 目录

                # 生成 noisy 文件路径: output_dir / id_
                noisy_file = os.path.join(output_dir, id_)

                # 混合信号
                mix = signal_by_db(clean_data, noise_data, snr)
                noisy_data = np.asarray(mix, dtype=np.int16)

                # 保存 noisy 文件
                sf.write(noisy_file, noisy_data, fs)
                print(f"  - 生成文件: {noisy_file}")

    print("--- 带噪语音数据生成完毕 ---")

    # 返回文件名 ID 列表
    clean_wav_ids = [os.path.split(f)[-1] if '/' in f else f for f in test_clean_files]
    return path_eval_noisy, noises, snrs, clean_wav_ids


if __name__ == "__main__":
    para = hparams()
    generate_evaluation_data(para)