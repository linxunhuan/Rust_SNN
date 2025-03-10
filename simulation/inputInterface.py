#!/Users/alessio/anaconda3/bin/python3

import numpy as np


def imgToSpikeTrain(image, dt, trainingSteps, inputIntensity, rng):

    ''' 
    使用泊松方法将黑白图像转换为 spikes 序列

    输入参数：

        1) image: 包含每个像素值的 NumPy 数组，像素值以整数表示
        2) dt: 时间步长，以毫秒为单位
        3) trainingSteps: 与像素spikes序列相关的总时间步数
        4) inputIntensity: 像素强度的当前值
        5) rng: NumPy 随机数生成器

    输出：
        二维布尔型 NumPy 数组。每行对应一个时间步，每列对应一个像素
    '''

    # 创建二维随机值数组
    random2D = rng.uniform(size = (trainingSteps, image.shape[0]))

    # 将图像转换为spikes序列
    return poisson(image, dt, random2D, inputIntensity)





def poisson(image, dt, random2D, inputIntensity):
    ''' 
    将像素的数值通过泊松过程转换为 spikes 序列。

    输入参数：
        1) image: 包含每个像素值的 NumPy 数组，像素值以整数表示
        2) dt: 时间步长，以毫秒为单位
        3) random2D: 包含像素最小值和最大值之间随机值的二维 NumPy 数组
        4) inputIntensity: 像素强度的最大频率（Hz）

    输出：
        包含每个像素 spikes 序列的布尔型二维数组
    '''
    # 将 dt 从毫秒转换为秒
    dt = dt * 1e-3

    # 归一化像素值（0-255 映射到 0-1）
    normalized_image = image / 255.0

    # 计算每像素的频率（Hz），基于输入强度
    max_frequency = inputIntensity  # 最大频率，例如 100 Hz
    frequency = normalized_image * max_frequency

    # 计算每步的脉冲概率
    spike_prob = 1.0 - np.exp(-frequency * dt)  # 泊松过程概率公式

    # 使用随机值生成脉冲
    return spike_prob > random2D