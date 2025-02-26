import numpy as np

def computePerformance(currentIndex, updateInterval, countersEvolution, 
                      labels, assignments, accuracies):
    '''
    计算网络的性能

    输入参数：
        1) currentIndex: 当前图像的索引，表示当前处理到了第几张图片
        2) updateInterval: 更新间隔，即每隔多少张图像计算一次性能
        3) countersEvolution: 二维 NumPy 数组，记录了过去 "updateInterval" 个周期内spikes计数器的历史
           - 每行对应一个训练步骤
           - 每列对应输出层中的一个元素（神经元）
        4) labels: NumPy 数组，包含所有图像的真实标签
        5) assignments: NumPy 数组，为每个输出神经元分配一个标签，用于分类映射
        6) accuracies: 字符串列表，记录历史的准确率

    输出：
        accuracies: 更新后的字符串列表，包含历史的准确率

    函数逻辑：
        该函数在指定的更新间隔检查网络性能，通过分析spikes计数器来推测分类结果，并与真实标签对比
    '''

    # 检查是否到达更新间隔的末尾
    # currentIndex % updateInterval == 0 表示当前索引是更新间隔的倍数
    # currentIndex > 0 避免在第一次循环就触发计算
    if currentIndex % updateInterval == 0 and currentIndex > 0:
        
        # 初始化最大spikes计数数组，长度为 updateInterval，初始值为 0
        # 用于跟踪每个时间步的最大spikes数
        maxCount = np.zeros(updateInterval)

        # 初始化分类结果数组，默认值设为 -1，表示未分类
        # dtype=np.int32 确保整数类型
        classification = -1 * np.ones(updateInterval, dtype=np.int32)

        # 获取当前更新间隔内的真实标签序列
        # 从 labels 中提取最近 updateInterval 个标签
        labelsSequence = labels[currentIndex - updateInterval : currentIndex]

        # 遍历所有可能的标签（0 到 9，对应 MNIST 的 10 个类别）
        for label in range(10):
            
            # 计算与当前标签关联的神经元的spikes总数
            # assignments == label 找到分配给当前标签的神经元
            # countersEvolution[:, assignments == label] 提取这些神经元的spikes数据
            # np.sum(..., axis=1) 按行求和，得到每个时间步的总spikes数
            spikesCount = np.sum(countersEvolution[:, assignments == label], axis=1)

            # 找到spikes计数大于当前最大值的时间步
            # whereMaxSpikes 是一个布尔数组，标记哪些时间步的spikes数超过了 maxCount
            whereMaxSpikes = spikesCount > maxCount

            # 将这些时间步的分类结果更新为当前标签
            classification[whereMaxSpikes] = label

            # 更新最大spikes计数，仅更新超过当前最大值的时间步
            maxCount[whereMaxSpikes] = spikesCount[whereMaxSpikes]

        # 打印计算出的分类结果，便于调试和观察
        print(classification)

        # 计算准确率并更新 accuracies 列表
        accuracies = updateAccuracy(classification, labelsSequence, accuracies)

    # 返回更新后的准确率历史
    return accuracies




def updateAccuracy(classification, labelsSequence, accuracies):
    '''
    计算准确率并将其添加到准确率历史列表中。

    输入参数：
        1) classification: NumPy 数组，记录了过去 "updateInterval" 个周期内网络的分类结果
        2) labelsSequence: NumPy 数组，记录了过去 "updateInterval" 个周期内的真实标签
        3) accuracies: 字符串列表，记录历史的准确率

    输出：
        accuracies: 更新后的字符串列表，包含历史的准确率

    函数逻辑：
        通过比较分类结果和真实标签，计算正确率并格式化为百分比字符串
    '''

    # 计算分类结果与真实标签相符的时间步数量
    # np.where(classification == labelsSequence) 返回相符位置的索引
    # [0].size 获取这些位置的数量
    correct = np.where(classification == labelsSequence)[0].size

    # 计算准确率百分比并格式化为两位小数字符串
    # correct/classification.size 计算正确比例
    # *100 转换为百分比
    # "{:.2f}".format() 保留两位小数
    accuracies += ["{:.2f}".format(correct/classification.size*100) + "%"]
    
    # 生成并打印准确率字符串，便于观察
    accuracyString = "\nAccuracy: " + str(accuracies) + "\n"
    print(accuracyString)

    # 返回更新后的准确率历史
    return accuracies