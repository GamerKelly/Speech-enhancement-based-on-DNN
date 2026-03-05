import os
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import pandas as pd
import warnings

# 忽略 PESQ 在某些情况下可能发出的警告
warnings.filterwarnings("ignore", ".*Did not find a matching filterbank.*")

# --- 配置 ---
clean_dir = '/data1/BOX/eval_clean/'
denoised_dir = '/data1/BOX/eval_noisy/babble/5'
OUTPUT_CSV = 'Babbble_5_noisy.csv'  # 输出文件名

# 如果增强后的文件名 (denoised) 比纯净文件名 (clean) 多一个前缀，请在此设置
# 示例: clean_file 是 'abc.wav'，denoised_file 是 'enh-abc.wav'
# ENHANCEMENT_PREFIX 应设置为 'enh-'
ENHANCEMENT_PREFIX = ''

# 音频的采样率，PESQ和STOI计算需要知道
SAMPLING_RATE = 16000  # 假设是16kHz，请根据您的数据调整
# PESQ 模式：'wb' (Wideband, 16kHz) 或 'nb' (Narrowband, 8kHz)。
# 由于 SAMPLING_RATE=16000，我们使用 'wb' 模式。
PESQ_MODE = 'wb'


def compute_scores_for_file(ref_path, deg_path, fs):
    """
    计算给定参考文件和退化文件的 PESQ 和 STOI 分数。
    """
    try:
        # 读取音频文件
        ref_signal, ref_fs = sf.read(ref_path)
        deg_signal, deg_fs = sf.read(deg_path)

        # 检查采样率是否匹配配置
        if ref_fs != fs or deg_fs != fs:
            print(f"警告: 文件 {os.path.basename(ref_path)} 或 {os.path.basename(deg_path)} 的采样率 ({ref_fs}/{deg_fs} Hz) 与配置 ({fs} Hz) 不匹配。")
            return None, None

        # 确保两个信号长度一致（对于语音增强评估通常是必需的）
        min_len = min(len(ref_signal), len(deg_signal))
        ref_signal = ref_signal[:min_len]
        deg_signal = deg_signal[:min_len]

        # --- 计算 PESQ ---
        # PESQ 函数需要两个参数：采样率、参考信号、退化信号、模式
        pesq_score = pesq(fs, ref_signal, deg_signal, PESQ_MODE)

        # --- 计算 STOI ---
        stoi_score = stoi(ref_signal, deg_signal, fs)

        return pesq_score, stoi_score

    except Exception as e:
        print(f"处理文件时发生错误 ({os.path.basename(ref_path)} 和 {os.path.basename(deg_path)}): {e}")
        return None, None


def batch_compute_and_save(clean_dir, denoised_dir, fs, output_csv, prefix, pesq_mode):
    """
    批量计算并保存结果到 CSV 文件
    """
    results_list = []
    print(f"---  开始批量计算和 CSV 保存 (PESQ mode: {pesq_mode}) ---")

    # 获取纯净语音文件夹中的所有.wav文件
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])

    if not clean_files:
        print(f"错误: 文件夹 `{clean_dir}` 中未找到 .wav 文件。")
        return

    for filename in clean_files:
        ref_path = os.path.join(clean_dir, filename)

        # 增强文件名 = 前缀 + 纯净文件名
        enh_filename = prefix + filename
        deg_path = os.path.join(denoised_dir, enh_filename)

        print(f"正在处理: Clean: {filename} -> Denoised: {enh_filename}...")

        if os.path.exists(deg_path):
            pesq_score, stoi_score = compute_scores_for_file(ref_path, deg_path, fs)

            if pesq_score is not None:
                results_list.append({
                    'Filename': filename,
                    'PESQ': f"{pesq_score:.4f}",
                    'STOI': f"{stoi_score:.4f}"
                })
            else:
                results_list.append({
                    'Filename': filename,
                    'PESQ': 'N/A',
                    'STOI': 'N/A'
                })
        else:
            print(f" 错误: 文件 {enh_filename} 在 `{denoised_dir}` 中未找到，跳过。")
            results_list.append({
                'Filename': filename,
                'PESQ': 'N/A',
                'STOI': 'N/A'
            })

    # --- 结果保存和平均分计算 ---
    if results_list:
        df = pd.DataFrame(results_list)

        # 计算平均分
        valid_pesq = pd.to_numeric(df['PESQ'], errors='coerce').dropna()
        valid_stoi = pd.to_numeric(df['STOI'], errors='coerce').dropna()
        avg_pesq = valid_pesq.mean() if not valid_pesq.empty else 'N/A'
        avg_stoi = valid_stoi.mean() if not valid_stoi.empty else 'N/A'

        average_row = pd.DataFrame([
            {'Filename': '--- Average Score ---',
             'PESQ': f"{avg_pesq:.4f}" if isinstance(avg_pesq, float) else avg_pesq,
             'STOI': f"{avg_stoi:.4f}" if isinstance(avg_stoi, float) else avg_stoi}
        ])
        # 合并结果和平均分行
        df = pd.concat([df, average_row], ignore_index=True)

        # 保存到 CSV
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n **成功!** 结果已保存到文件: `{output_csv}`")
        print(f"  平均 PESQ: {average_row['PESQ'].iloc[0]}")
        print(f"  平均 STOI: {average_row['STOI'].iloc[0]}")
    else:
        print("\n 没有成功处理任何文件，未生成 CSV 文件。")

if __name__ == '__main__':
    batch_compute_and_save(
        clean_dir=clean_dir,
        denoised_dir=denoised_dir,
        fs=SAMPLING_RATE,
        output_csv=OUTPUT_CSV,
        prefix=ENHANCEMENT_PREFIX,
        pesq_mode=PESQ_MODE
    )

