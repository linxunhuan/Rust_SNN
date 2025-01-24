use crate::snn::neuron::Neuron;

/* * LIF子模块 * */

/**
表示 LIF 模型中的神经元的对象
 */
#[derive(Debug)]
pub struct LifNeuron {
    /* const fields */
    v_th:    f64,       /* 阈值电位，神经元达到这个电位时会触发发放（尖峰） */
    v_rest:  f64,       /* 静息电位，神经元在没有受到任何输入时的电位 */
    v_reset: f64,       /* 复位电位，神经元在发放后会重置到这个电位 */
    tau:     f64,       /* 时间常数，描述了神经元膜电位的变化速度 */
    dt:      f64,       /* 两个连续时刻之间的时间间隔，用于数值模拟时的步长 */
    /* mutable fields */
    v_mem:   f64,       /* 膜电位，当前时刻神经元的电位 */
    ts:      u64,       /* 最后一个瞬间，至少接收到一个尖峰的时间戳 */
}

impl LifNeuron {
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64, dt: f64) -> Self {
        Self {
            v_th,
            v_rest,
            v_reset,
            tau,
            dt,
            v_mem: v_rest,
            ts: 0u64,       // 初始化 ts 字段为 0
        }
    }

    /* 神经元对象参数的获取器 */
    pub fn get_v_th(&self) -> f64 {
        self.v_th
    }

    pub fn get_v_rest(&self) -> f64 {
        self.v_rest
    }

    pub fn get_v_reset(&self) -> f64 {
        self.v_reset
    }

    pub fn get_tau(&self) -> f64 {
        self.tau
    }

    pub fn get_dt(&self) -> f64 { self.dt }

    pub fn get_v_mem(&self) -> f64 {
        self.v_mem
    }

    pub fn get_ts(&self) -> u64 {
        self.ts
    }

}

impl Neuron for LifNeuron {
    /*
        当神经元接收到至少一个 spike 时，这个函数会更新神经元的膜电位
    */
    fn compute_v_mem(&mut self, t: u64, extra_weighted_sum: f64, intra_weighted_sum: f64) -> u8 {
        let weighted_sum = extra_weighted_sum +    /* 正贡献 */
            intra_weighted_sum;      /* 负贡献 */

        /* 使用 LIF 公式计算神经元膜电位 */
        let exponent = -(((t - self.ts) as f64) * self.dt / self.tau);
        self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * exponent.exp() + weighted_sum;

        /* 更新 ts - 最后一个瞬间至少接收到一个正 spike */
        self.ts = t;

        return if self.v_mem > self.v_th {
            /* 重置膜电位 */
            self.v_mem = self.v_reset;
            1   /* 发放 spike */
        } else {
            0   /* 不发放 spike */
        };
    }

    fn initialize(&mut self) {
        self.v_mem = self.v_rest;
        self.ts = 0u64;
    }
}


/*
    为 LifNeuron 对象实现特征
*/
impl Clone for LifNeuron {
    fn clone(&self) -> Self {
        Self {
            v_th:    self.v_th,
            v_rest:  self.v_rest,
            v_reset: self.v_reset,
            tau:     self.tau,
            dt:      self.dt,
            v_mem:   self.v_mem,
            ts:      self.ts
        }
    }
}

unsafe impl Send for LifNeuron {}

