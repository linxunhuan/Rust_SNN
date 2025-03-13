use std::slice::IterMut;
use std::sync::{Arc, Mutex};
use crate::snn::layer::Layer;
use crate::snn::neuron::Neuron;
use crate::snn::processor::Processor;
use crate::snn::SpikeEvent;

/* * 脉冲神经网络结构 * */

/**
    表示脉冲神经网络（Spiking Neural Network, SNN）对象本身
    - N: 表示神经元实现的泛型类型
    - NET_INPUT_DIM: 网络的输入维度，即输入层的大小
    - NET_OUTPUT_DIM: 网络的输出维度，即输出层的大小
    具有像 NET_INPUT_DIM 这样的泛型常量类型，可以在编译时检查用户提供的输入大小
 */
#[derive(Debug)]
pub struct SNN<N: Neuron + Clone + Send + 'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize> {
    layers: Vec<Arc<Mutex<Layer<N>>>>,   /* 神经网络的层 */
}

impl<N: Neuron + Clone + Send + 'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize>
SNN<N, NET_INPUT_DIM, NET_OUTPUT_DIM> {
    pub fn new(layers: Vec<Arc<Mutex<Layer<N>>>>) -> Self {
        Self {
            layers
        }
    }

    /* Getters for the SNN object */
    pub fn get_layers_number(&self) -> usize {
        self.layers.len()
    }

    pub fn get_layers(&self) -> Vec<Layer<N>> {
        self.layers.iter().map(|layer| layer.lock().unwrap().clone()).collect()
    }

    /** 实际通过脉冲神经网络处理输入的 spike 并产生相应的输出 spike 
        - 'spikes' 包含一个针对每个输入层神经元的二进制数组，每个数组的 spike 数量相同
        等于输入的持续时间（spikes 是一个 0/1 的矩阵， 每个输入神经元一行，每个时间点一列）
    该方法能够在编译时检查用户输入
        例如: snn.process(&[[0,1,1], [1,0,1]]) /* 输入层有 2 个神经元，每个接收 3 个 spike */ */
    pub fn process<const SPIKES_DURATION: usize>(&mut self, spikes: &[[u8; SPIKES_DURATION]; NET_INPUT_DIM])
                                                 -> [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] {
        /* 将 spikes 编码为 SpikeEvent(s) */
        let input_spike_events = SNN::<N, NET_INPUT_DIM, NET_OUTPUT_DIM>::encode_spikes(spikes);

        /* 处理输入并生成 SNN 的输出 spikes */
        let processor = Processor {};
        let output_spike_events = processor.process_events(self, input_spike_events);

        /* 将输出解码为数组形状 */
        let decoded_output: [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] =
            SNN::<N, NET_INPUT_DIM, NET_OUTPUT_DIM>::decode_spikes(output_spike_events);

        decoded_output
    }

/**
    (与 process() 方法相同，但在 *run-time* 检查输入 spike 的大小：
    spikes 必须有一个数量等于 NET_INPUT_DIM 的 Vec，所有这些 Vec 的长度必须相同，否则 panic!())
 */
pub fn process_dyn(&mut self, spikes: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    /* 检查 spikes vec 的数量 */
    if spikes.len() != NET_INPUT_DIM {
        panic!("错误：维度不匹配 - 每个输入层的神经元必须有自己的 spikes vec");
    }

    /* 将输入 spikes 编码为 spike 事件 */

    let mut spikes_events = Vec::<SpikeEvent>::new();
    let mut spikes_duration: Option<usize> = None;

    for (n, neuron_spikes) in spikes.into_iter().enumerate() {
        let temp_len = neuron_spikes.len();

        /* 检查 spikes 的持续时间 - 它们必须具有相同的大小 */
        match spikes_duration {
            None => spikes_duration = Some(temp_len),
            Some(duration) => if temp_len != duration {
                panic!("错误：发现不同大小的 spikes vec - 每个输入层的神经元的 spikes 必须具有相同的持续时间")
            }
        }

        if n == 0 {    /* 第一个循环... */
            /* ...创建所有的 spike 事件 */
            (0..spikes_duration.unwrap()).for_each(|t| {
                let spike_event = SpikeEvent::new(t as u64, Vec::<u8>::new());
                spikes_events.push(spike_event);
            });
        }

        /* 将每个 spike 复制到 spike_events vec 中 */
        for t in 0..spikes_duration.unwrap() {
            let temp_spike = neuron_spikes[t];
            if temp_spike != 0 && temp_spike != 1 {
                panic!("错误：输入 spike 必须为 0 或 1，对于神经元 {} 在 t={}", n, t);
            }

            spikes_events[t].spikes.push(temp_spike);
        }
    }

    /* 运行 SNN */
    let processor = Processor {};
    let output_spike_events = processor.process_events(self, spikes_events);

    /* 解码输出 spike 事件 */

    /* 创建和初始化输出对象 */
    let mut output_spikes: Vec<Vec<u8>> = Vec::new();

    for _ in &output_spike_events.get(0)
        .unwrap_or(&SpikeEvent::new(0, Vec::<u8>::new()))
        .spikes {
        /* 创建与第一个输出 spike 事件的长度（输出神经元的数量）相同数量的内部 Vec<u8> */
        output_spikes.push(vec![0u8; spikes_duration.unwrap()]);
    }

    /* 将处理后的 spikes 复制到输出 spikes vec 中 */
    for spike_event in output_spike_events {
        for (out_neuron_index, spike) in spike_event.spikes.into_iter().enumerate() {
            output_spikes[out_neuron_index][spike_event.ts as usize] = spike;
        }
    }

    output_spikes
}


    /* private functions */

    /**
        此函数将输入脉冲矩阵（0/1）编码为 SpikeEvents 的 Vec
    */
    fn encode_spikes<const SPIKES_DURATION: usize>(spikes: &[[u8; SPIKES_DURATION]; NET_INPUT_DIM])
        -> Vec<SpikeEvent> {
        
        let mut spike_events = Vec::<SpikeEvent>::new();  /* 创建一个空的 SpikeEvent 向量 */
        
        for t in 0..SPIKES_DURATION {
            let mut t_spikes = Vec::<u8>::new();  /* 创建一个空的 Vec<u8> 用于存储当前时间点的 spikes */

            /* 获取每个神经元的输入 spikes */
            for in_neuron_index in 0..NET_INPUT_DIM {
                if spikes[in_neuron_index][t] != 0 && spikes[in_neuron_index][t] != 1 {
                    panic!("错误：输入 spike 必须为 0 或 1");  /* 检查 spike 是否为 0 或 1 */
                }
                t_spikes.push(spikes[in_neuron_index][t]);  /* 将 spike 加入当前时间点的 spikes 向量中 */
            }

            let t_spike_event = SpikeEvent::new(t as u64, t_spikes);  /* 创建一个新的 SpikeEvent */
        spike_events.push(t_spike_event);  /* 将 SpikeEvent 加入 spike_events 向量中 */
        }
        spike_events  /* 返回编码后的 SpikeEvent 向量 */
    }


    /**
        这个函数解码一个包含 SpikeEvents 的 Vec 并返回一个由 1/0 组成的输出 spikes 矩阵，
        每行对应一个输出神经元
    */
    fn decode_spikes<const SPIKES_DURATION: usize>(spikes: Vec<SpikeEvent>)
        -> [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] {

        /* 创建一个全为 0 的矩阵，大小为 NET_OUTPUT_DIM 行、SPIKES_DURATION 列 */
        let mut raw_spikes = [[0u8; SPIKES_DURATION]; NET_OUTPUT_DIM];  

        for spike_event in spikes {

            /* 遍历 spike_event 中的每个 spike */  
            for (out_neuron_index, spike) in spike_event.spikes.into_iter().enumerate() {  
                raw_spikes[out_neuron_index][spike_event.ts as usize] = spike;
            }
        }

        raw_spikes  /* 返回解码后的 spike 矩阵 */
    }
}

impl<'a, N: Neuron + Clone + Send + 'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize>
IntoIterator for &'a mut SNN<N, NET_INPUT_DIM, NET_OUTPUT_DIM> {
    type Item = &'a mut Arc<Mutex<Layer<N>>>;
    type IntoIter = IterMut<'a, Arc<Mutex<Layer<N>>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter_mut()
    }
}
