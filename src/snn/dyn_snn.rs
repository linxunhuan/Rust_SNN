use std::slice::IterMut;
use std::sync::{Arc, Mutex};
use crate::neuron::Neuron;
use crate::snn::layer::Layer;
use crate::snn::processor::Processor;
use crate::SpikeEvent;

/* * 动态脉冲神经网络结构 * */

/**
    表示（动态）脉冲神经网络本身的对象
    - N: 是一个泛型类型，表示神经元的实现

    这里没有定义类似于NET_INPUT_DIM, NET_OUTPUT_DIM等的泛型常量变量，因为所有的计算和检查都是在*运行时*进行的。
*/
#[derive(Debug, Clone)]
pub struct DynSNN<N: Neuron + Clone + 'static> {
    // 一个包含多个层的向量，每一层都由Arc和Mutex保护，以确保线程安全
    layers: Vec<Arc<Mutex<Layer<N>>>>
}

impl<N: Neuron + Clone> DynSNN<N> {
    // 创建一个新的DynSNN对象
    pub fn new(layers: Vec<Arc<Mutex<Layer<N>>>>) -> Self {
        Self { layers }
    }

    /* Getters */
    pub fn get_layers_number(&self) -> usize {
        self.layers.len()
    }

    fn get_input_layer_dimension(&self) -> usize {
        let first_layer = self.layers[0].lock().unwrap();
        let input_layer_dimension = first_layer.get_weights().first().unwrap().len();

        input_layer_dimension
    }

    fn get_output_layer_dimension(&self) -> usize {
        let last_layer = self.layers.last().unwrap().lock().unwrap();
        let output_dimension = last_layer.get_neurons_number();

        output_dimension
    }

    pub fn get_layers(&self) -> Vec<Layer<N>> {
        self.layers.iter().map(|layer| layer.lock().unwrap().clone()).collect()
    }

    /**
        实际上通过脉冲神经网络处理输入脉冲并产生相应的输出脉冲。
        'spikes' 包含每个输入层神经元的数组，并且每个数组有相同数量的脉冲，等于输入的持续时间
        (spikes是一个矩阵，每个输入神经元占一行，每个时间点占一列)
        这个方法在 *运行时* 检查用户输入
    */
    pub fn process(&mut self, spikes: &Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        // * 检查并计算脉冲的持续时间 *
        let spikes_duration = self.compute_spikes_duration(spikes);

        // 获取输入层和输出层的维度
        let input_layer_dimension = self.get_input_layer_dimension();
        let output_layer_dimension = self.get_output_layer_dimension();

        // * 将脉冲编码为SpikeEvent(s) *
        let input_spike_events =
            DynSNN::<N>::encode_spikes(input_layer_dimension, spikes, spikes_duration);

        // * 处理输入 *
        let processor = Processor{};
        let output_spike_events = processor.process_events(self, input_spike_events);

        // * 将输出解码为数组形式 *
        let decoded_output = DynSNN::<N>::decode_spikes(output_layer_dimension,
                                                         output_spike_events, spikes_duration);

        // 返回解码后的输出
        decoded_output
    }


    /**
        这个函数检查传递到'spikes'中的每个向量是否有相同数量的脉冲。
        如果相同，它返回脉冲的持续时间；否则，会触发一个错误。
     */
    fn compute_spikes_duration(&self, spikes: &Vec<Vec<u8>>) -> usize {
        // 计算第一个Vec的长度（如果不存在，则为0）
        let spikes_duration = spikes.get(0)
                                            .unwrap_or(&Vec::new())
                                            .len();

        // 遍历每个神经元的脉冲
        for neuron_spikes in spikes {
            // 检查每个神经元的脉冲长度是否与第一个脉冲长度相同
            if neuron_spikes.len() != spikes_duration {
                panic!("每个神经元的脉冲持续时间必须相等");
            }
        }
        // 返回脉冲持续时间
        spikes_duration
    }


    /** 
        这个函数将接收到的输入脉冲编码为一个Vec<SpikeEvent>以处理它们。
    */
    fn encode_spikes(input_layer_dimension: usize, spikes: &Vec<Vec<u8>>, spikes_duration: usize) -> Vec<SpikeEvent> {
        // 创建一个空的Vec<SpikeEvent>用于存储编码后的脉冲事件
        let mut spike_events = Vec::<SpikeEvent>::new();
    
        // 检查输入脉冲的数量是否与输入层的维度一致
        if spikes.len() != input_layer_dimension {
            panic!("输入脉冲的数量与输入层的维度不一致： 'spikes'必须为每个神经元提供一个Vec");
        }

        // 遍历整个脉冲持续时间
        for t in 0..spikes_duration {
            // 创建一个空的Vec<u8>用于存储时间点t的脉冲
            let mut t_spikes = Vec::<u8>::new();

            // 检索每个神经元在时间点t的输入脉冲
            for in_neuron_index in 0..spikes.len() {
                // 检查输入脉冲是否为0或1
                if spikes[in_neuron_index][t] != 0 && spikes[in_neuron_index][t] != 1 {
                    panic!("错误：输入脉冲在N={}处和t={}时必须为0或1", in_neuron_index, t);
                }
                // 将脉冲加入到时间点t的脉冲Vec中
                t_spikes.push(spikes[in_neuron_index][t]);
            }

            // 创建一个新的SpikeEvent并加入到spike_events中
            let t_spike_event = SpikeEvent::new(t as u64, t_spikes);
            spike_events.push(t_spike_event);
        }

        // 返回编码后的脉冲事件
        spike_events
    }


    /**
        这个函数解码一个含有SpikeEvent的Vec，并返回一个由0和1组成的输出脉冲矩阵
     */
    fn decode_spikes(output_layer_dimension: usize, spikes: Vec<SpikeEvent>, spikes_duration: usize) -> Vec<Vec<u8>> {
        // 创建一个大小为output_layer_dimension x spikes_duration的二维向量raw_spikes，并初始化为0
        let mut raw_spikes = vec![vec![0; spikes_duration]; output_layer_dimension];

        for spike_event in spikes {
            // 遍历每一个SpikeEvent中的脉冲（spikes），并枚举其索引
            for (out_neuron_index, spike) in spike_event.spikes.into_iter().enumerate() {
                // 将脉冲值（spike）填入raw_spikes矩阵中相应的位置
                raw_spikes[out_neuron_index][spike_event.ts as usize] = spike;
            }
        }

        // 返回构建好的raw_spikes矩阵
        raw_spikes
    }

}

impl<'a, N: Neuron + Clone + 'static> IntoIterator for &'a mut DynSNN<N> {
    type Item = &'a mut Arc<Mutex<Layer<N>>>;
    type IntoIter = IterMut<'a, Arc<Mutex<Layer<N>>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter_mut()
    }
}
