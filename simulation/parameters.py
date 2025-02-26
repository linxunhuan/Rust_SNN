# 用于打开网络参数文件夹中包含的 .npy 文件的脚本

import numpy as np
import re

thresholdsFile = "./networkParameters/thresholds.npy"  # 阈值文件的路径
weightsFile = "./networkParameters/weights.npy"        # 权重文件的路径
thresholdsOut = "./networkParameters/thresholdsOut.txt"  # 阈值输出文件的路径
weightsOut = "./networkParameters/weightsOut.txt"        # 权重输出文件的路径

# 打开 thresholdsFile 并将其值写入 thresholdsOut.txt 文件
with open(thresholdsFile, 'rb') as fp:
    thresholds = np.load(fp)  # 从 .npy 文件加载阈值数据

with open(thresholdsOut, 'w') as f:
    for el in thresholds:      # 遍历阈值数组中的每个元素
        for el2 in el:         # 遍历每个子数组中的值
            f.write(str(el2) + '\n')  # 将值转换为字符串并逐行写入文件

# 打开 weightsFile 并将其值写入 weightsOut.txt 文件
with open(weightsFile, 'rb') as fp:
    weights = np.load(fp)  # 从 .npy 文件加载权重数据

with open(weightsOut, 'w') as f:
    for el in weights:         # 遍历权重数组中的每个元素
        string = str(el).replace('[', '')  # 移除字符串中的左括号
        string = string.replace(']', '')   # 移除字符串中的右括号
        string = string.replace('\n', ' ') # 将换行符替换为空格
        string = re.sub(' +', ' ', string) # 用单个空格替换多个连续空格
        f.write(string + '\n')  # 将处理后的字符串逐行写入文件