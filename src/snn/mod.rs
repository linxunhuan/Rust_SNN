
pub mod builders;
    mod layer; /* private */
pub mod neuron;
    mod dyn_snn;
    mod snn;
    mod processor;

/**
    表示单层生成的输出脉冲的对象
*/
#[derive(Debug)]
pub struct SpikeEvent {
    ts: u64,            /* 离散时间瞬间 */
    spikes: Vec<u8>,    /* 该瞬间的spick向量（每个输入神经元为 1/0）  */
}

impl SpikeEvent {
    pub fn new(ts: u64, spikes: Vec<u8>) -> Self {
        Self { ts, spikes }
    }
}
