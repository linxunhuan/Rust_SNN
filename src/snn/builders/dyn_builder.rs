/* * dyn_builder 子模块 * */

use std::sync::{Arc, Mutex};
use crate::neuron::Neuron;
use crate::snn::dyn_snn::DynSNN;
use crate::snn::layer::Layer;

/**
    包含描述DynSNN架构的配置参数的对象
*/
#[derive(Clone)]
pub struct DynSnnParams<N: Neuron> {
    pub input_dimensions: usize,            // 输入层的维度
    pub neurons: Vec<Vec<N>>,               // 每层的神经元
    pub extra_weights: Vec<Vec<Vec<f64>>>,  // 层之间的（正）权重
    pub intra_weights: Vec<Vec<Vec<f64>>>,  // 同一层内的（负）权重
    pub num_layers: usize,                  // 层的数量
}

/**
    用于配置和创建动态脉冲神经网络（DynSNN）的对象。
    允许通过传递所有网络参数（神经元、额外权重、内部权重等）到一个函数中来一次性配置网络。
    - 这里所有与网络维度相关的检查都是在*运行时*进行的。
*/
#[derive(Clone)]
pub struct DynSnnBuilder<N: Neuron> {
    // 包含描述DynSNN架构的配置参数的对象
    params: DynSnnParams<N>
}

impl<N: Neuron + Clone> DynSnnBuilder<N> {
    // 创建一个新的DynSnnBuilder对象
    pub fn new(input_dimension: usize) -> Self {
        Self {
            params: DynSnnParams {
                input_dimensions: input_dimension,  // 输入层的维度
                neurons: vec![],                    // 每层的神经元
                extra_weights: vec![],              // 层之间的（正）权重
                intra_weights: vec![],              // 同一层内的（负）权重
                num_layers: 0                       // 层的数量
            }
        }
    }
    // 获取当前配置参数的副本
    pub fn get_params(&self) -> DynSnnParams<N> {
        self.params.clone()
    }

    /**
        会进行所有与网络的内部权重相关的检查。
        - 检查神经元的数量是否等于内部权重矩阵的行数
        - 检查神经元的数量是否等于内部权重矩阵的列数
        - 检查内部权重的值是否都是负数，并且在[-1, 0]范围内
    */
    fn check_intra_weights(&self, num_neurons: usize, weights: &Vec<Vec<f64>>)  {
        // 检查神经元的数量是否等于内部权重矩阵的行数
        if num_neurons != weights.len() {
            panic!("神经元的数量必须等于内部权重矩阵的行数");
        }

        for row in weights {
            // 检查神经元的数量是否等于内部权重矩阵的列数
            if num_neurons != row.len() {
                panic!("神经元的数量必须等于内部权重矩阵的列数");
            }
            // 检查每个权重值是否为负数
            for weight in row {
                if *weight > 0.0 {
                    panic!("内部权重必须是负数");
                }
            }
        }
    }

    /**
        进行所有与网络的额外权重相关的检查。
        - 检查神经元的数量是否等于额外权重矩阵的行数
        - 检查神经元的数量是否等于额外权重矩阵的列数
        - 检查额外权重矩阵的列数是否等于前一层的神经元数量
        - 检查额外权重的值是否都是正数，并且在[0, 1]范围内
    */

    fn check_weights(&self, num_neurons: usize, weights: &Vec<Vec<f64>>) {
        // 检查神经元的数量是否等于权重矩阵的行数
        if num_neurons != weights.len() {
            panic!("神经元的数量必须等于权重矩阵的行数");
        }

        for row in weights {
            // 如果是第一个层，则检查行的长度是否等于输入层的维度
            if self.params.num_layers == 0 {
                if row.len() != self.params.input_dimensions {
                    panic!("神经元的数量必须等于权重矩阵的列数");
                }
            } else {
                // 否则，检查行的长度是否等于前一层的神经元数量
                if row.len() != self.params.neurons[self.params.num_layers - 1].len() {
                    panic!("权重矩阵中的列数必须等于前一层的神经元数量");
                }
            }
            // 检查每个权重值是否为正数
            for weight in row {
                if *weight < 0.0 {
                    panic!("权重必须是正数");
                }
            }
        }
    }


    /**
        通过指定所有请求的参数为网络添加一个新层
    */
    pub fn add_layer(self, neurons: Vec<N>, extra_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>) -> Self {
        // 检查内部权重
        self.check_intra_weights(neurons.len(), &intra_weights);
        // 检查额外权重
        self.check_weights(neurons.len(), &extra_weights);

        // 克隆当前的配置参数
        let mut params = self.params;

        // 将新层的神经元、额外权重和内部权重添加到参数中
        params.neurons.push(neurons);
        params.extra_weights.push(extra_weights);
        params.intra_weights.push(intra_weights);
        params.num_layers += 1;

        // 返回带有新参数的Self
        Self { params }
    }


/**
        通过指定所有请求的参数为网络添加一个新层。
        - 所有神经元具有相同的参数
    */
    pub fn add_layer_with_same_neurons(self, neuron: N, num_neurons: usize, extra_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>) -> Self {
        // 检查内部权重
        self.check_intra_weights(num_neurons, &intra_weights);
        // 检查额外权重
        self.check_weights(num_neurons, &extra_weights);

        // 克隆当前的配置参数
        let mut params = self.params;

        // 创建一个具有相同参数的神经元向量
        let mut neurons = Vec::with_capacity(num_neurons);

        for _i in 0..num_neurons {
            neurons.push(neuron.clone());
        }

        // 将神经元、额外权重和内部权重添加到参数中
        params.neurons.push(neurons);
        params.extra_weights.push(extra_weights);
        params.intra_weights.push(intra_weights);
        params.num_layers += 1;

        // 返回带有新参数的Self
        Self { params }
    }


/**
        根据已经定义的特性创建并初始化整个动态脉冲神经网络（DynSNN）。
        - 如果网络没有层，则会引发错误
    */
    pub fn build(self) -> DynSNN<N> {

        // 如果层的数量为0，则引发错误
        if self.params.num_layers == 0 {
            panic!("网络必须至少有一层");
        }

        // 检查神经元层的数量是否与权重层的数量一致
        if self.params.neurons.len() != self.params.extra_weights.len() ||
           self.params.neurons.len() != self.params.intra_weights.len() {
            /* 这种情况不应发生 */
            panic!("错误：神经元层的数量与权重层的数量不对应");
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

        // 创建并返回新的DynSNN
        DynSNN::new(layers)
    }
}