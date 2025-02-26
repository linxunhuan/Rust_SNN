#!/Users/alessio/anaconda3/bin/python3

import numpy as np

dictDecoder = {
    
    "8": "ubyte",	# 无符号字节
    "9": "byte",	# 有符号字节
    "11": ">i2",	# 大端序16位整数
    "12": ">i4",	# 大端序32位整数
    "13": ">f4",	# 大端序32位浮点数
    "14": ">f8"		# 大端序64位浮点数（双精度）
}

def loadDataset(images, labels):

    ''' 
    将 MNIST 数据集中的图像和标签加载到两个 NumPy 数组中

    输入参数：

        1) images: 字符串，对应包含 28x28 黑白图像的 MNIST 文件名，
        这些图像以 idx 格式存储

        2) labels: 字符串，对应包含标签的 MNIST 文件名，
        标签以无符号字节序列的形式存储

    返回值：

        1) imgArray: 二维 NumPy 数组，包含从 MNIST 文件中读取的所有图像
        每个图像存储为形状为 (1,784) 的 NumPy 数组，便于用作神经网络的输入，
        其中每个输入神经元对应一个像素

        2) labelsArray: NumPy 数组，包含从 MNIST 文件中读取的所有标签，
        以整数形式存储
    '''

    # 将图像文件的全部内容加载到内存缓冲区
    imgBuffer = readFile(images)

    # 创建用于训练/测试的图像数组
    imgArray = idxBufferToArray(imgBuffer)

    # 将标签文件的全部内容加载到内存缓冲区
    labelsBuffer = readFile(labels)

    # 创建用于训练/测试的标签数组
    labelsArray = idxBufferToArray(labelsBuffer)

    return imgArray, labelsArray


def readFile(filename):

    '''
    读取二进制文件的全部内容并将其存储在内存缓冲区中

    输入参数：
        filename: 字符串，对应要读取的文件名

    返回值：
        readData: 缓冲区，其中存储了文件的全部内容，以字节序列形式表示
    '''
    with open(filename, "r+b") as f:
        readData = f.read()

    return readData


def idxBufferToArray(buffer):

    '''
    将以 idx 格式存储数据的二进制缓冲区转换为 NumPy 数组

    输入参数：
        buffer: 以 idx 格式编码数据的字节序列
        可以通过调用 readFile() 函数从 idx 文件中读取数据获得

    返回值：
        data: NumPy 数组，仅包含从 idx 缓冲区中提取的数据

    magic number和维度信息会被读取并用于确定输出数组的形状，但它们不会出现在输出数据数组中
    '''

    # 读取并解码magic number
    dtype, dataDim = magicNumber(buffer)

    # magic number占4字节，每个维度占4字节
    offset = 4*dataDim+4

    # 读取并存储数据结构的所有维度
    dimensions = readDimensions(buffer, dataDim)

    # 将数据存储到适当形状的 NumPy 数组中
    data = loadData(buffer, dtype, offset, dimensions)

    return data


def magicNumber(buffer):

    '''
    读取并解码magic number

    输入参数：
        buffer: 以 idx 格式编码数据的字节序列
        可以通过调用 readFile() 函数从 idx 文件中读取数据获得

    返回值：
        1) dtype: 表示数据类型的字符串。详情见 dictDecoder
        2) dataDim: 存储在 idx 缓冲区中数据结构的维度数量

    注意，decodeDataType 期望输入为十进制数，而数据类型在magic number中以十六进制值编码
    幸运的是，NumPy 的 frombuffer 函数会直接将十六进制值转换为对应的十进制值
    '''

    # 将magic number读取为四个单独的字节
    mn = np.frombuffer(buffer, dtype="ubyte", count=4)

    # 解码数据类型字节
    dtype = decodeDataType(mn[2])

    # 解码数据维度字节
    dataDim = mn[3]

    return dtype, dataDim


def decodeDataType(intCode):

    '''
    解码magic number中编码数据类型的字节。

    输入参数：
        intCode: 从magic number的第二低有效字节读取的整数值，以十进制格式表示

    返回值：

        函数返回一个字符串，对应magic number中编码的数据类型。详情见 dictDecoder
    '''
    return dictDecoder[str(intCode)]


def readDimensions(buffer, dataDim):

    '''
    从 idx 缓冲区中读取适当数量的维度。维度以32位整数存储
    输入参数：
        1) buffer: 以 idx 格式编码数据的字节序列。
        可以通过调用 readFile() 函数从 idx 文件中读取数据获得
        2) dataDim: 要读取的维度数量。可以通过调用 magicNumber() 函数获得

    返回值：
        函数返回一个形状为 (1, dataDim) 的 NumPy 数组，包含所有维度
    '''
    return np.frombuffer(buffer, dtype=">u4", count=dataDim, offset=4)


def loadData(buffer, dtype, offset, dimensions):

    '''
    从以 idx 格式存储数据的缓冲区中读取完整的数据结构

    输入参数：
        1) buffer: 以 idx 格式编码数据的字节序列
        可以通过调用 readFile() 函数从 idx 文件中读取数据获得
        2) dtype: 编码要读取数据类型的字符串
        3) offset: 表示数据在 idx 缓冲区中起始位置的偏移量，以字节为单位
        4) dimensions: 包含要读取数据结构所有维度的 NumPy 数组

    返回值：
        函数返回一个具有适当形状的 NumPy 数组，包含所有数据
    '''

    # 将数据存储为单一维度的 NumPy 数组
    data = np.frombuffer(buffer, dtype=dtype, count=np.prod(dimensions),
        offset=offset) 

    # 将数据重塑为多维 NumPy 数组并返回
    return reshapeData(data, dimensions)



def reshapeData(data, dimensions):

    '''
    重塑 NumPy 数组

    新形状通过解释从 idx 缓冲区读取的维度获得

    如果缓冲区只包含单一维度，则意味着数据以单一数组形式存储，
    输出 NumPy 数组也将具有单一维度

    如果维度是多个，则输出数组将被重塑为二维 NumPy 数组，
    将高阶维度展平为单一维度

    输入参数：
        1) data: 包含所有数据的 NumPy 数组
        2) dimensions: 包含所有维度的 NumPy 数组

    输出值：
        data: 重塑后的 NumPy 数组
    '''
    
    # 仅在必要时重塑
    if dimensions.size > 1:

        # 重塑为二维数组
        arrayDim = np.prod(dimensions[1:])
        data = np.reshape(data, (dimensions[0], arrayDim))

    return data