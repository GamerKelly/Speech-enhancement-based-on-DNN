import os

def create_train_scp_pure_filename(data_root_dir, output_scp_path, file_extension='.wav'):
    """
    遍历数据集根目录（包括子目录），查找所有音频文件，并将它们的纯文件名
    写入 train.scp 文件。

    参数:
    data_root_dir (str): 纯净语音数据集的根目录。
    output_scp_path (str): 生成的 .scp 文件的路径和名称，例如 'scp/train.scp'。
    file_extension (str): 要查找的文件扩展名 (默认是 '.wav')。
    """

    if not os.path.isdir(data_root_dir):
        print(f"错误：数据集目录 '{data_root_dir}' 不存在。请检查路径。")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_scp_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录：{output_dir}")

    print(f"开始扫描目录：{data_root_dir}")
    file_names = []

    # 遍历 data_root_dir 中的所有子目录
    for root, _, files in os.walk(data_root_dir):
        for file in files:
            # 检查文件是否是我们需要的扩展名
            if file.endswith(file_extension):
                # 仅添加纯文件名 (例如: 'file1.wav')
                # 无论文件位于哪个子目录，都只记录其名称
                file_names.append(file)

    # 写入 .scp 文件
    try:
        with open(output_scp_path, 'w', encoding='utf-8') as f:
            for name in file_names:
                f.write(name + '\n')

        print(f"成功生成 .scp 文件：{output_scp_path}")
        print(f"共找到 {len(file_names)} 个 {file_extension} 文件。")
        print("注意：.scp 文件中仅包含纯文件名，不包含目录结构。")

    except IOError as e:
        print(f"写入文件时发生错误：{e}")


# --- 配置您的路径 ---
DATASET_ROOT_DIR = "/data1/BOX/train_clean/"
OUTPUT_SCP_FILE = "scp/train_clean.scp"
FILE_EXT = ".wav"

# 执行函数
if __name__ == "__main__":
    create_train_scp_pure_filename(
        data_root_dir=DATASET_ROOT_DIR,
        output_scp_path=OUTPUT_SCP_FILE,
        file_extension=FILE_EXT
    )
