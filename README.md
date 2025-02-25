# 脉冲神经网络库
- [脉冲神经网络库](#脉冲神经网络库)
  - [描述](#描述)
  - [依赖](#依赖)
  - [仓库结构](#仓库结构)
  - [组织](#组织)
  - [主要结构](#主要结构)
  - [主要方法](#主要方法)
  - [使用示例](#使用示例)

## 描述
这是一个用 Rust 编写的库，旨在建模 脉冲神经网络

该库支持在脉冲数据集上执行 脉冲神经网络 模型的实现。它不支持网络的训练阶段，仅支持执行阶段。

## 依赖
- `Rust` (version 1.56.1)
- `Cargo` (version 1.56.0)

## 仓库结构
仓库的结构如下：
- `src/` 包含库的源代码
  + `bin/`    包含演示脚本
  + `models/` 包含具体模型的实现（此处仅为 Lif Neuron）
  + `snn/`    包含 SNN 的通用实现
    + `builders` 包含 SNN 的构建器对象
- `tests/` 包含库的测试

## 组织
该库的组织结构如下：

- ### 构建器
  Builder 模块允许你实际创建网络结构，包括相应的层、每层的神经元、它们之间的权重以及同一层神经元之间的权重
  该库提供了两种 Builder 实现：
  - #### SnnBuilder 

    SnnBuilder 允许静态创建 脉冲神经网络，为每一层提供静态的神经元向量、权重和层内权重
    该库可以在编译时检查网络结构的正确性，但这意味着网络的所有结构都分配在**栈（Stack）**上（不适合大型网络）

  - #### DynSnnBuilder 
   DynSnnBuilder 允许动态创建 脉冲神经网络，为每一层提供动态的神经元向量、权重和层内权重
   该库只能在运行时检查网络结构的正确性，但这意味着网络的所有结构都分配在**堆（Heap）**上（适合大型网络）

- ### Network
  Network 模块允许你在给定输入上实际执行网络
  该库提供了两种 Network 实现：
  - #### Snn 

    Snn 由 SnnBuilder 创建，并允许通过 process() 方法在给定输入上执行网络
    与 SnnBuilder 一样，Snn 接收静态的脉冲向量作为输入，并同样产生静态的脉冲向量作为输出

  - #### DynSnn

    DynSnn 由 DynSnnBuilder 创建，并允许通过 process() 方法在给定输入上执行网络
    与 DynSnnBuilder 一样，DynSnn 接收动态的脉冲向量作为输入，并同样产生动态的脉冲向量作为输出
    输入的正确性只能在运行时检查

## 主要结构
该库提供了以下主要结构：

- LifNeuron 表示 泄漏积分-发放 模型的神经元，可用于构建神经元 Layer

```rust
pub struct LifNeuron {
    /* 常量字段 */
    v_th:    f64,       /* 阈值电位 */
    v_rest:  f64,       /* 静息电位 */
    v_reset: f64,       /* 重置电位 */
    tau:     f64, 
    dt:      f64,       /* 两个连续时刻之间的时间间隔 */
    
    /* 可变字段 */
    v_mem:   f64,       /* 膜电位 */
    ts:      u64,       /* 上次接收到至少一个脉冲的时刻 */
}
```
有关 泄漏积分-发放 模型的更多信息，请参见 [此处](https://www.nature.com/articles/s41598-017-07418-y).

- `Layer` 表示一层神经元，可用于构建由多层组成的 SNN 或 DynSNN
```rust
pub struct Layer<N: Neuron + Clone + Send + 'static> {
    neurons: Vec<N>,                /* 该层的神经元 */
    weights: Vec<Vec<f64>>,         /* 该层与前一层神经元之间的权重 */
    intra_weights: Vec<Vec<f64>>,   /* 该层内部神经元之间的权重 */
    prev_output_spikes: Vec<u8>     /* 前一时刻的输出脉冲 */
}
```

- `SpikeEvent` 表示某时刻神经元层发放的事件。它封装了网络中流动的脉冲
```rust
pub struct SpikeEvent {
    ts: u64,            /* 离散时间点 */
    spikes: Vec<u8>,    /* 该时刻的脉冲向量（每个输入神经元为 1/0） */
}
```

- `SNN` 表示由多个 `Layer` 向量组成的 `脉冲神经网络`.
```rust
pub struct SNN<N: Neuron + Clone + Send + 'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize> {
    layers: Vec<Arc<Mutex<Layer<N>>>>
}
```

- `DynSNN` 表示由多个 `Layer` 向量组成的 `动态脉冲神经网络`.
```rust
pub struct DynSNN <N: Neuron + Clone + 'static>{
    layers: Vec<Arc<Mutex<Layer<N>>>>
}
```

- `Processor` 是负责管理层线程并处理输入脉冲事件的对象
```rust
pub struct Processor { }
```

- `SnnBuilder` 表示 `SNN` 的构建器
```rust
pub struct SnnBuilder<N: Neuron + Clone + Send + 'static> {
    params: SnnParams<N>
}

pub struct SnnParams<N: Neuron + Clone + Send + 'static> {
    pub neurons: Vec<Vec<N>>,               /* 每个层的神经元 */
    pub extra_weights: Vec<Vec<Vec<f64>>>,  /* 层之间的（正）权重 */
    pub intra_weights: Vec<Vec<Vec<f64>>>,  /* 同一层内的（负）权重 */
}
```

-  `DynSnnBuilder` 表示 `DynSNN` 的构建器
```rust
pub struct DynSnnBuilder<N: Neuron> {
    params: DynSnnParams<N>
}

pub struct DynSnnParams<N: Neuron> {
    pub input_dimensions: usize,            /* 网络输入层的维度 */
    pub neurons: Vec<Vec<N>>,               /* 每个层的神经元 */
    pub extra_weights: Vec<Vec<Vec<f64>>>,  /* 层之间的（正）权重 */
    pub intra_weights: Vec<Vec<Vec<f64>>>,  /* 同一层内的（负）权重 */
    pub num_layers: usize,                  /* 层数 */
}
```

## 主要方法
该库提供了以下主要方法：
 - ### 构建器方法
   - #### `SnnBuilder` 方法:
   
     - **new()** 
     
        ```rust
          pub fn new() -> Self
         ```         
       
        创建一个新的 `SnnBuilder` 
     
     - **add_layer()** 
     
       ```rust
          pub fn add_layer(self) -> WeightsBuilder<N, OUTPUT_DIM, NET_INPUT_DIM> 
       ``` 
         向 `SnnBuilder` 添加一个新的（空）层
     
     -  **weights()**
     
        ```rust
          pub fn weights<const NUM_NEURONS: usize>(mut self, weights: [[f64; INPUT_DIM]; NUM_NEURONS])
                                           -> NeuronsBuilder<N, NUM_NEURONS, NET_INPUT_DIM>
          ```
          向当前层添加与前一层的权重
     
       - **neurons()** method:
       
         ```rust
             pub fn neurons(mut self, neurons: [N; NUM_NEURONS]) -> IntraWeightsBuilder<N, NUM_NEURONS, NET_INPUT_DIM>
         ```
          向当前层添加神经元
     - **intra_weights()** method
        ```rust
         pub fn intra_weights(mut self, intra_weights: [[f64; NUM_NEURONS]; NUM_NEURONS])
                    -> LayerBuilder<N, NUM_NEURONS, NET_INPUT_DIM>
         ```
       
         向当前层添加层内权重

     - **build()** method:
     
        ```rust
         pub fn build(self) -> SNN<N, { NET_INPUT_DIM }, { OUTPUT_DIM }>
         ```
       
         根据 `SnnBuilder` 收集的信息构建 `SNN`

 
- #### `DynSnnBuilder` :
   - **new()** :
   
     ```rust
     pub fn new(input_dimension: usize) -> Self 
        ```
     
     创建一个新的 `DynSnnBuilder`
   - **add_layer()** :
   
     ```rust
     pub fn add_layer(self, neurons: Vec<N>, extra_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>) -> Self
     ```
     
     使用给定的 `neurons`、`weights` 和 `intra_weights` 参数向 SNN 添加一个新 `层`

   - **build()** :
   
     ```rust
     pub fn build(self) -> DynSNN<N>
     ```
     
     根据 `DynSnnBuilder` 收集的信息构建 `DynSNN`

 - ### Network Methods
   - #### `Snn` method:
      - process() :
      
        ```rust
         pub fn process<const SPIKES_DURATION: usize>(&mut self, spikes: &[[u8; SPIKES_DURATION]; NET_INPUT_DIM])
                                                 -> [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM]
        ```
        
        处理作为参数传递的输入脉冲，并返回网络的输出脉冲
   - #### `DynSnn` method:
        - process() :
        
            ```rust
             pub fn process(&mut self, spikes: &Vec<Vec<u8>>)
                                                 -> Vec<Vec<u8>> 
            ```
          
            处理作为参数传递的输入脉冲，并返回网络的输出脉冲
   

## 使用示例
以下示例展示了如何使用 SnnBuilder 静态创建一个具有 2 个输入神经元和一层 3 个 LifNeuron 的 脉冲神经网络，并在每个神经元 3 个时刻的给定输入上执行它
```rust
use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;


 let mut snn = SnnBuilder::new()
        .add_layer()    /* 第一层（输入维度自动推断） */
            .weights([
                [0.1, 0.2],     /* 从输入层到第 1 个神经元的权重 */
                [0.3, 0.4],     /* 从输入层到第 2 个神经元的权重 */
                [0.5, 0.6]      /* 从输入层到第 3 个神经元的权重 */
            ]).neurons([    
                /* 3 个 LIF 神经元 */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ]).intra_weights([
                [0.0, -0.1, -0.15],     /* 从同一层到第 1 个神经元的权重 */
                [-0.05, 0.0, -0.1],     /* 从同一层到第 2 个神经元的权重 */
                [-0.15, -0.1, 0.0]      /* 从同一层到第 3 个神经元的权重 */
        ])
        .add_layer()    /* 第二层 */
            .weights([
                [0.11, 0.29, 0.3],      /* 从前一层到第 1 个神经元的权重 */
                [0.33, 0.41, 0.57]      /* 从前一层到第 2 个神经元的权重 */
            ]).neurons([    
                /* 2 个 LIF 神经元 */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ]).intra_weights([
                [0.0, -0.25],       /* 从同一层到第 1 个神经元的权重 */
                [-0.10, 0.0]        /* 从同一层到第 2 个神经元的权重 */
        ]).build();     /* 创建网络 */
    
    /* 处理输入脉冲 */
    let output_spikes = snn.process(&[
        [1,0,1],    /* 第 1 个神经元输入 */
        [0,0,1]     /* 第 2 个神经元输入 */
    ]);       
```
以下示例展示了如何使用 DynSnnBuilder 动态创建一个具有 2 个输入神经元和一层 3 个 LifNeuron 的 脉冲神经网络，并在每个神经元 3 个时刻的给定输入上执行它

```rust
use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;


 let mut snn = SnnBuilder::new()
        .add_layer()    /* 第一层（输入维度自动推断） */
            .weights([
                [0.1, 0.2],     /* 从输入层到第 1 个神经元的权重 */
                [0.3, 0.4],     /* 从输入层到第 2 个神经元的权重 */
                [0.5, 0.6]      /* 从输入层到第 3 个神经元的权重 */
            ]).neurons([    
                /* 3 个 LIF 神经元 */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ]).intra_weights([
                [0.0, -0.1, -0.15],     /* 从同一层到第 1 个神经元的权重 */
                [-0.05, 0.0, -0.1],     /* 从同一层到第 2 个神经元的权重 */
                [-0.15, -0.1, 0.0]      /* 从同一层到第 3 个神经元的权重 */
        ])
        .add_layer()    /* 第二层 */
            .weights([
                [0.11, 0.29, 0.3],      /* 从前一层到第 1 个神经元的权重 */
                [0.33, 0.41, 0.57]      /* 从前一层到第 2 个神经元的权重 */
            ]).neurons([    
                /* 2 个 LIF 神经元 */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ]).intra_weights([
                [0.0, -0.25],       /* 从同一层到第 1 个神经元的权重 */
                [-0.10, 0.0]        /* 从同一层到第 2 个神经元的权重 */
        ]).build();     /* 创建网络 */
    
    /* 处理输入脉冲 */
    let output_spikes = snn.process(&[
        [1,0,1],    /* 第 1 个神经元输入 */
        [0,0,1]     /* 第 2 个神经元输入 */
    ]);    
```
