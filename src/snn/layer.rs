/* * 私有层子模块 * */

use crate::snn::neuron::Neuron;
use crate::snn::SpikeEvent;
use std::sync::mpsc::{Receiver, Sender};

/* 表示脉冲神经网络层的对象 */
#[derive(Debug)]
/// 表示包含神经元的层的结构体
pub struct Layer<N: Neuron + Clone + Send + 'static> {
    neurons: Vec<N>,              /* 这一层的神经元 */
    weights: Vec<Vec<f64>>,       /* 这一层神经元与上一层神经元之间的权重 */
    intra_weights: Vec<Vec<f64>>, /* 这一层神经元之间的权重 */
    prev_output_spikes: Vec<u8>,  /* 前一个时刻的输出 spikes */
}

impl<N: Neuron + Clone + Send + 'static> Layer<N> {
    pub fn new(neurons: Vec<N>, weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>) -> Self {
        let num_neurons = neurons.len();
        Self {
            neurons,
            weights,
            intra_weights,
            prev_output_spikes: vec![0; num_neurons],
        }
    }

    pub fn get_neurons_number(&self) -> usize {
        self.neurons.len()
    }

    pub fn get_neurons(&self) -> Vec<N> {
        self.neurons.clone()
    }

    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

    pub fn get_intra_weights(&self) -> Vec<Vec<f64>> {
        self.intra_weights.clone()
    }

    /** 处理来自上一层的输出 SpikeEvent，根据网络中的神经元模型，
    并将产生的 spikes 发送到下一层。
    - layer_input_rc: 是一个来自上一层的通道接收器
    - layer_output_tx: 是一个发送到下一网络层（或如果是输出层则发送到 SNN 本身）的通道发送器 */
    pub fn process(
        &mut self,
        layer_input_rc: Receiver<SpikeEvent>,
        layer_output_tx: Sender<SpikeEvent>,
    ) {
        /* 初始化数据结构，以便 SNN 可以重复使用 */
        self.initialize();

        /* 监听来自上一层的 SpikeEvent 并处理它们 */
        while let Ok(input_spike_event) = layer_input_rc.recv() {
            let instant = input_spike_event.ts; /* 输入 spike 到达的时间点 */
            let mut output_spikes = Vec::<u8>::with_capacity(self.neurons.len()); /* 存储当前层的输出 spikes */
            let mut at_least_one_spike = false; /* 检查是否至少有一个神经元发放了 spike */

            /*
                对每个神经元计算 intra 和 extra 加权和，
                然后获取输出 spike
            */
            for (index, neuron) in self.neurons.iter_mut().enumerate() {
                let mut extra_weighted_sum = 0f64;
                let mut intra_weighted_sum = 0f64;

                /* 计算 extra 加权和 */
                let extra_weights_pairs = self.weights[index]
                    .iter()
                    .zip(input_spike_event.spikes.iter());

                for (weight, spike) in extra_weights_pairs {
                    if *spike != 0 {
                        extra_weighted_sum += *weight;
                    }
                }

                /* 计算 intra 加权和
                (intra_weights[index] 包含了到当前神经元的连接权重) */
                let intra_weights_pairs = self.intra_weights[index]
                    .iter()
                    .zip(self.prev_output_spikes.iter());

                for (i, (weight, spike)) in intra_weights_pairs.enumerate() {
                    /* 忽略反射性连接 */
                    if i != index && *spike != 0 {
                        intra_weighted_sum += *weight;
                    }
                }

                /* 计算膜电位并确定神经元是否发放 spike */
                let neuron_spike =
                    neuron.compute_v_mem(instant, extra_weighted_sum, intra_weighted_sum);
                output_spikes.push(neuron_spike);

                if !at_least_one_spike && neuron_spike == 1u8 {
                    at_least_one_spike = true;
                }
            }

            /* 保存输出 spikes 以供后续使用 */
            self.prev_output_spikes = output_spikes.clone();

            /* 检查是否至少有一个神经元发放了 spike - 如果没有，则不发送任何 spike */
            if !at_least_one_spike {
                continue;
            }
            /* 至少有一个神经元发放了 spike -> 将输出 spikes 发送到下一层 */

            let output_spike_event = SpikeEvent::new(instant, output_spikes);

            layer_output_tx
                .send(output_spike_event)
                .expect(&format!("发送输入 spike 事件 t={} 时出现意外错误", instant));
        }

        /*
            我们不需要手动释放发送器，因为当层超出作用域时，它会自动释放
        */
    }

    fn initialize(&mut self) {
        self.prev_output_spikes.clear(); /* 重置 prev_output_spikes */
        self.neurons
            .iter_mut()
            .for_each(|neuron| neuron.initialize()); /* 重置神经元 */
    }
}

/*
    Layer 对象的特征实现
*/

impl<N: Neuron + Clone + Send + 'static> Clone for Layer<N> {
    fn clone(&self) -> Self {
        Self {
            neurons: self.neurons.clone(),
            weights: self.weights.clone(),
            intra_weights: self.intra_weights.clone(),
            prev_output_spikes: self.prev_output_spikes.clone(),
        }
    }
}

unsafe impl<N: Neuron + Clone> Sync for Layer<N> {}
