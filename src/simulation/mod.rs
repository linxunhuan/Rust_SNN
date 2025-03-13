pub mod inputInterface; // 输入接口模块
pub mod mnist; // MNIST 数据集处理模块
pub mod outputInterface; // 输出接口模块

// 导出关键函数以供 main.rs 使用
pub use inputInterface::img_to_spike_train; // 将图像转换为脉冲序列
pub use mnist::load_dataset; // 加载 MNIST 数据集
pub use outputInterface::compute_performance; // 计算性能指标
