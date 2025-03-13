/* * 构建器子模块 * */

use std::sync::{Arc, Mutex};
use crate::snn::layer::Layer;
use crate::snn::neuron::Neuron;
use crate::snn::snn::SNN;

/**
    包含描述SNN架构的配置参数的对象
    - *neurons*: 每一层的神经元向量
    - *extra_weights*: 每一层的权重矩阵（每个矩阵包含该层中每个神经元的向量）
    - *intra_weights*: 每一层的权重矩阵（每个矩阵包含该层中每个神经元的向量）
 */
#[derive(Debug, Clone)]
pub struct SnnParams<N: Neuron + Clone + Send + 'static> {
    pub neurons: Vec<Vec<N>>,               // 每层的神经元向量
    pub extra_weights: Vec<Vec<Vec<f64>>>,  // 层之间的（正）权重矩阵
    pub intra_weights: Vec<Vec<Vec<f64>>>,  // 同一层内的（负）权重矩阵
}

/**
    这个对象用于配置和创建脉冲神经网络（Spiking Neural Network, SNN）
    允许一步一步地**静态**配置网络，每次添加一层，
    指定每一层的（额外）权重、神经元和内部权重。
    在编译时检查输入参数的维度是否正确。
    - 遵循（流式的）Builder设计模式。
 */
#[derive(Debug, Clone)]
pub struct SnnBuilder<N: Neuron + Clone + Send + 'static> {
    // 包含描述SNN架构的配置参数的对象
    params: SnnParams<N>
}

impl<N: Neuron + Clone + Send + 'static> SnnBuilder<N> {
    // 创建一个新的SnnBuilder对象
    pub fn new() -> Self {
        Self {
            params: SnnParams {
                neurons: vec![],               // 每层的神经元向量
                extra_weights: vec![],         // 层之间的（正）权重矩阵
                intra_weights: vec![]          // 同一层内的（负）权重矩阵
            }
        }
    }

    // 获取当前配置参数的副本
    pub fn get_params(&self) -> SnnParams<N> {
        self.params.clone()
    }

    /* (网络的*输入维度*可以由编译器自动推断) */
    pub fn add_layer<const INPUT_DIM: usize>(self) -> WeightsBuilder<N, INPUT_DIM, INPUT_DIM> {
        WeightsBuilder::<N, INPUT_DIM, INPUT_DIM>::new(self.params)
    }
}


/* ** 流式建造者模式结构体 ** */

/* * 权重 * */
/**
    - INPUT_DIM: 是新层的输入维度
    - NET_INPUT_DIM: 是整个神经网络的输入维度
 */
#[derive(Debug, Clone)]
pub struct WeightsBuilder<N: Neuron + Clone + Send + 'static, const INPUT_DIM: usize, const NET_INPUT_DIM: usize> {
    // 包含描述SNN架构的配置参数的对象
    params: SnnParams<N>
}


impl<N: Neuron + Clone + Send + 'static, const INPUT_DIM: usize, const NET_INPUT_DIM: usize>
    WeightsBuilder<N, INPUT_DIM, NET_INPUT_DIM> {
    
    // 创建一个新的WeightsBuilder对象
    pub fn new(params: SnnParams<N>) -> Self {
        Self { params }
    }

    // 获取当前配置参数的副本
    pub fn get_params(&self) -> SnnParams<N> {
        self.params.clone()
    }

    /**
        指定前一层和新层之间连接的权重。
        接收每层神经元的一个数组，包含神经元与前一层神经元的所有有序连接权重
    */
    pub fn weights<const NUM_NEURONS: usize>(mut self, weights: [[f64; INPUT_DIM]; NUM_NEURONS])
                                         -> NeuronsBuilder<N, NUM_NEURONS, NET_INPUT_DIM> {
        // 创建一个权重的Vec<Vec<f64>>
        let mut weights_vec: Vec<Vec<f64>> = Vec::new();

        /* 将数组参数转换为Vec */
        for neuron_weights in &weights {
            neuron_weights.iter().for_each(|w| 
                if w < &0.0 {
                    panic!("权重必须是正数");
                }
            );
            weights_vec.push(Vec::from(neuron_weights.as_slice()));
        }

        /* 保存层的权重 */
        self.params.extra_weights.push(weights_vec);
        NeuronsBuilder::<N, NUM_NEURONS, NET_INPUT_DIM>::new(self.params)
    }
}


/* * 神经元 * */
#[derive(Debug, Clone)]
pub struct NeuronsBuilder<N: Neuron + Clone + Send + 'static, const NUM_NEURONS: usize, const NET_INPUT_DIM: usize> {
    params: SnnParams<N>
}

impl<N: Neuron + Clone + Send + 'static, const NUM_NEURONS: usize, const NET_INPUT_DIM: usize>
    NeuronsBuilder<N, NUM_NEURONS, NET_INPUT_DIM> {
    
    // 创建一个新的NeuronsBuilder对象
    pub fn new(params: SnnParams<N>) -> Self {
        Self { params }
    }

    // 获取当前配置参数的副本
    pub fn get_params(&self) -> SnnParams<N> {
        self.params.clone()
    }

    /**
        将一个有序神经元数组添加到层中
     */
    pub fn neurons(mut self, neurons: [N; NUM_NEURONS]) -> IntraWeightsBuilder<N, NUM_NEURONS, NET_INPUT_DIM> {
        // 将神经元数组转换为Vec并添加到参数中的neurons向量中
        self.params.neurons.push(Vec::from(neurons));
        // 创建并返回一个新的IntraWeightsBuilder对象
        IntraWeightsBuilder::<N, NUM_NEURONS, NET_INPUT_DIM>::new(self.params)
    }

    /**
        将一个有序神经元数组添加到层中
        - 所有神经元具有相同的参数
    */
    pub fn neurons_with_same_parameters(mut self, neuron: N, num_neurons: usize) -> IntraWeightsBuilder<N, NUM_NEURONS, NET_INPUT_DIM> {
        // 创建一个具有相同参数的神经元向量
        let mut neurons: Vec<N> = Vec::with_capacity(num_neurons);

        for _i in 0..num_neurons {
            neurons.push(neuron.clone());
        }

        // 将神经元向量添加到参数中的neurons向量中
        self.params.neurons.push(neurons);
        // 创建并返回一个新的IntraWeightsBuilder对象
        IntraWeightsBuilder::<N, NUM_NEURONS, NET_INPUT_DIM>::new(self.params)
    }
}


/* * 内部权重 * */
#[derive(Debug, Clone)]
pub struct IntraWeightsBuilder<N: Neuron + Clone + Send + 'static, const NUM_NEURONS: usize, const NET_INPUT_DIM: usize> {
    // 包含描述SNN架构的配置参数的对象
    params: SnnParams<N>
}

impl<N: Neuron + Clone + Send + 'static, const NUM_NEURONS: usize, const NET_INPUT_DIM: usize>
    IntraWeightsBuilder<N, NUM_NEURONS, NET_INPUT_DIM> {
    
    // 创建一个新的IntraWeightsBuilder对象
    pub fn new(params: SnnParams<N>) -> Self {
        Self { params }
    }

    // 获取当前配置参数的副本
    pub fn get_params(&self) -> SnnParams<N> {
        self.params.clone()
    }


    /**
        指定了同一层神经元之间连接的（负）权重
        接收一个类似矩阵的参数，每个神经元都有一个数组。每个数组包含从其同胞（兄弟神经元）到该神经元的有序连接权重
        
        注意：数组中对应于神经元自身连接的元素将被忽略（可以设为0）。
        例如，在一个有3个神经元的层中，一个内部权重矩阵的示例可以是： 
        [[0, -0.1, -0.3], [-0.2, 0, -0.7], [-0.9, -0.4, 0]]
        第x个数组中的第y个元素表示从神经元Y到神经元X的连接权重
        因此，神经元#1到神经元#0的连接权重为-0.1； 而神经元#0到神经元#2的连接权重为-0.9
     */
    pub fn intra_weights(mut self, intra_weights: [[f64; NUM_NEURONS]; NUM_NEURONS])
                    -> LayerBuilder<N, NUM_NEURONS, NET_INPUT_DIM> {
        // 创建一个空的Vec<Vec<f64>>用于存储内部权重
        let mut intra_weights_vec: Vec<Vec<f64>> = Vec::new();

        /* 将数组形式的内部权重参数转换为Vec */
        for neuron_intra_weights in &intra_weights {
            neuron_intra_weights.iter().for_each(|w| if w > &0f64 {
                panic!("内部权重必须是负数");
            });
            // 将每个神经元的内部权重转换为Vec并添加到intra_weights_vec中
            intra_weights_vec.push(Vec::from(neuron_intra_weights.as_slice()));
        }

        /* 保存层的内部权重 */
        self.params.intra_weights.push(intra_weights_vec);
        // 创建并返回一个新的LayerBuilder对象
        LayerBuilder::<N, NUM_NEURONS, NET_INPUT_DIM>::new(self.params)
    }
}

/* * 层 * */
/**
    允许添加新层，或者构建并获取具有迄今为止定义的特征的 SNN
*/
#[derive(Debug, Clone)]
pub struct LayerBuilder<N: Neuron + Clone + Send + 'static, const OUTPUT_DIM: usize, const NET_INPUT_DIM: usize> {
    // 包含描述SNN架构的配置参数的对象
    params: SnnParams<N>
}

impl<N: Neuron + Clone + Send + 'static, const OUTPUT_DIM: usize, const NET_INPUT_DIM: usize>
    LayerBuilder<N, OUTPUT_DIM, NET_INPUT_DIM> {
    
    // 创建一个新的LayerBuilder对象
    pub fn new(params: SnnParams<N>) -> Self {
        Self { params }
    }

    // 获取当前配置参数的副本
    pub fn get_params(&self) -> SnnParams<N> {
        self.params.clone()
    }

    /**
        为SNN添加一个新层
     */
    pub fn add_layer(self) -> WeightsBuilder<N, OUTPUT_DIM, NET_INPUT_DIM> {
        WeightsBuilder::<N, OUTPUT_DIM, NET_INPUT_DIM>::new(self.params)
    }

    /**
        根据已经定义的特性创建并初始化整个脉冲神经网络（SNN）。
     */
    pub fn build(self) -> SNN<N, {NET_INPUT_DIM}, {OUTPUT_DIM}> {
        // 检查神经元层的数量是否与权重层的数量一致
        if self.params.neurons.len() != self.params.extra_weights.len() ||
           self.params.neurons.len() != self.params.intra_weights.len() {
            /* 这种情况不应发生 */
            panic!("错误：神经元层的数量与权重层的数量不对应")
        }

        // 创建一个空的Vec<Arc<Mutex<Layer<N>>>>用于存储层
        let mut layers: Vec<Arc<Mutex<Layer<N>>>> = Vec::new();

        // 创建迭代器用于遍历神经元、额外权重和内部权重
        let mut neurons_iter = self.params.neurons.into_iter();
        let mut extra_weights_iter = self.params.extra_weights.into_iter();
        let mut intra_weights_iter = self.params.intra_weights.into_iter();

        /* 检索每层的神经元、额外权重和内部权重 */
        while let Some(layer_neurons) = neurons_iter.next() {
            let layer_extra_weights = extra_weights_iter.next().unwrap();
            let layer_intra_weights = intra_weights_iter.next().unwrap();

            /* 创建并保存新层 */
            let new_layer = Layer::new(layer_neurons, layer_extra_weights, layer_intra_weights);
            layers.push(Arc::new(Mutex::new(new_layer)));
        }

        // 创建并返回新的SNN
        SNN::<N, NET_INPUT_DIM, OUTPUT_DIM>::new(layers)
    }
}
