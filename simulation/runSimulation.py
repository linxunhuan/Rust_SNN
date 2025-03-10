import numpy as np

# 导入自定义模块，用于加载 MNIST 数据集、生成spikes序列和计算性能
from mnist import loadDataset
from inputInterface import imgToSpikeTrain
from outputInterface import computePerformance

# 时间步长，以毫秒为单位
dt = 5.0

# spikes序列的持续时间，以毫秒为单位
trainDuration = 350

# 计算步数 = spikes序列持续时间除以时间步长，结果取整
computationSteps = int(trainDuration / dt)

# 输入像素值的归一化强度，用于调整spikes生成的频率
inputIntensity = 400

# 每隔多少张图像评估一次准确率
updateInterval = 100

# 网络结构定义
N_layers = 1          # 网络层数，这里是单层网络
N_neurons = [400]     # 每层神经元数量，这里输出层有 400 个神经元
N_inputs = 784        # 输入层大小，对应 MNIST 图像的 28x28=784 个像素

# 创建 NumPy 默认随机数生成器，用于生成随机spikes
rng = np.random.default_rng()

# MNIST 测试数据集文件路径
images = "./mnist/t10k-images-idx3-ubyte"  # 测试图像文件
labels = "./mnist/t10k-labels-idx1-ubyte"  # 测试标签文件

# 包含输出层每个神经元对应标签的文件
assignmentsFile = "./networkParameters/assignments.npy"

# 输入spikes和输出计数器的输出文件路径
inputSpikesFilename = "inputSpikes.txt"
outputCountersFilename = "outputCounters.txt"

# 初始化准确率历史列表
accuracies = []

# 初始化spikes计数器历史数组
# 形状为 (updateInterval, N_neurons[-1])，即 (100, 400)，记录最近 100 次迭代的输出计数
countersEvolution = np.zeros((updateInterval, N_neurons[-1]))

# 从文件中加载神经元标签分配
with open(assignmentsFile, 'rb') as fp:
    assignments = np.load(fp)  # 加载 .npy 文件，形状应为 (400,)，每个元素是一个标签

# 导入 MNIST 测试数据集
imgArray, labelsArray = loadDataset(images, labels)
# imgArray: 图像数组，形状为 (10000, 784)，每行是一个展平的 28x28 图像
# labelsArray: 标签数组，形状为 (10000,)，每个元素是 0-9 的整数

# 总循环次数，这里设置为 301 次（可能只处理数据集的一部分）
numberOfCycles = 301

# 遍历整个数据集的核心循环
for i in range(numberOfCycles):
    
    # 打印当前迭代次数（从 1 开始计数）
    print("\nIteration: ", i+1)

    # 将当前图像的像素值转换为spikes序列
    # imgArray[i] 是第 i 张图像，形状为 (784,)
    # 输出 spikesTrains 形状为 (computationSteps, 784)，即 (3500, 784)
    spikesTrains = imgToSpikeTrain(imgArray[i], dt, computationSteps, inputIntensity, rng)

    # ----------------------------------------------------------------------
    # 将 NumPy 数组写入文件，格式可以自由选择。
    # 数组有 3500 行（每行对应一个时间步），784 列（每列对应一个输入）。
    
    with open(inputSpikesFilename, "w") as fp:
        for step in spikesTrains:  # 遍历每个时间步的spikes数据
            # 将布尔数组转换为整数列表，去掉首尾括号，移除逗号和空格，写入文件
            fp.write(str(list(step.astype(int)))[1:-1].replace(",", "").replace(" ", ""))
            fp.write("\n")  # 每行一个时间步的spikes序列
    # 结果文件每行是 784 个 0 或 1，表示该时间步每个像素是否产生spikes
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # 调用 Rust 脚本处理spikes数据
    #
    import subprocess as sp

    rustScript = "../target/debug/pds_snn"  # Rust 可执行文件路径

    # 调用 Rust 程序，假设它会读取 inputSpikes.txt 并生成 outputCounters.txt
    sp.run(rustScript)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # 从文件中读取输出计数器并转换为 NumPy 数组。
    # 这里先初始化一个零向量，稍后用文件内容填充。
    outputCounters = np.zeros(N_neurons[-1]).astype(int)  # 形状为 (400,) 的整数数组

    with open(outputCountersFilename, "r") as fp:
        j = 0
        for line in fp:  # 逐行读取文件
            outputCounters[j] = int(line)  # 将每行转换为整数，填充到数组
            j += 1
    # outputCounters 保存当前图像处理后输出层每个神经元的spikes计数
    # ----------------------------------------------------------------------

    # 将当前输出计数器存入历史数组，使用模运算实现循环覆盖
    countersEvolution[i % updateInterval] = outputCounters

    # 计算网络性能并更新准确率历史
    accuracies = computePerformance(i, updateInterval, countersEvolution, labelsArray, assignments, accuracies)