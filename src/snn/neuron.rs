/* * 神经元子模块 * */

/**
    用于实现所有神经元模型的特征
    它表示一个层中的一般神经元
*/
pub trait Neuron: Send {
    /** 只有当从上一层到达了一些输入spikes时，才会调用神经元函数；
        因此，extra_weighted_sum 始终大于 0
        - t: 输入spikes到达时的时间点
        - extra_weighted_sum: *输入spikes* 与 *传入权重* 的点积
        - intra_weighted_sum: 前一个瞬间的 *输入spikes* 与 *层内权重* 的点积，
                              其中至少一个神经元（上一层的）发放了spikes
    */
    fn compute_v_mem(&mut self, t: u64, extra_weighted_sum: f64, intra_weighted_sum: f64) -> u8;

    /**
        将神经元恢复到初始状态：初始化所有数据结构
     */
    fn initialize(&mut self);
}
