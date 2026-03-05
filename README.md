# Speech-enhancement-based-on-DNN
基于DNN模型，分别使用Mapping和Masking两种方式来实现语音增强。

## 关于DNN-Mapping的使用说明 ##
在本项目中，纯净语音数据来自于TIMIT数据集中的TrainDatas集，噪声数据来自于Noise92数据集中的babble和white噪声，带噪语音数据是这两者混合后得到的结果。训练前请准备好纯净语音数据以及噪声数据。

- 运行get_train_clean_scp.py，生成train_clean.scp文件（需要该文件来合成带噪语音信号）
- 运行generate_training_data.py，生成带噪语音信号以及train_DNN_data.scp文件。
- 运行dataset.py，检验训练用数据集是否正确
- 运行train.py，进行模型训练
