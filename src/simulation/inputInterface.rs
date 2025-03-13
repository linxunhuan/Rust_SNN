use ndarray::{Array1, Array2};
use rand::Rng;

/// 将图像转换为脉冲序列，使用泊松方法。
///
/// # 输入参数
/// - `image`: 一维图像数组，像素值为 0-255 的灰度值，类型为 u8。
/// - `dt`: 时间步长，单位为毫秒。
/// - `training_steps`: 训练步数（时间步总数）。
/// - `input_intensity`: 输入强度。
/// - `rng`: 随机数生成器。
///
/// # 输出
/// 二维布尔数组，形状为 (training_steps, image.len())，表示脉冲序列。
/// 每行对应一个时间步，每列对应一个像素。
pub fn img_to_spike_train(
    image: &Array1<u8>,
    dt: f64,
    training_steps: usize,
    input_intensity: f64,
    rng: &mut impl Rng,
) -> Array2<bool> {
    // 生成二维随机数组，形状为 (training_steps, image.len())，值在 [0, 1) 之间
    let random_2d =
        Array2::from_shape_fn((training_steps, image.len()), |_| rng.gen_range(0.0..1.0));

    // 将图像转换为脉冲序列
    poisson(image, dt, &random_2d, input_intensity)
}

/// 泊松方法，将图像像素值转换为脉冲序列。
///
/// # 输入参数
/// - `image`: 一维图像数组，像素值为 0-255 的灰度值，类型为 u8。
/// - `dt`: 时间步长，单位为毫秒。
/// - `random_2d`: 二维随机数组，形状为 (training_steps, image.len())，值在 [0, 1) 之间。
/// - `input_intensity`: 输入强度。
///
/// # 输出
/// 二维布尔数组，形状为 (training_steps, image.len())，表示脉冲序列。
fn poisson(
    image: &Array1<u8>,
    dt: f64,
    random_2d: &Array2<f64>,
    input_intensity: f64,
) -> Array2<bool> {
    // 将 dt 从毫秒转换为秒
    let dt = dt * 1e-3;

    // 将图像数组转换为 f64 类型并计算缩放值
    let image_f64 = image.mapv(|x| x as f64);
    let scaled_image = image_f64 * (input_intensity / 8.0) * dt;

    // 广播并转换为拥有数组
    let broadcasted = scaled_image.broadcast(random_2d.raw_dim()).unwrap();
    let broadcasted_owned = broadcasted.to_owned(); // 转换为拥有数组，元素为 f64

    // 逐元素比较
    let spikes = Array2::from_shape_fn(random_2d.dim(), |(i, j)| {
        broadcasted_owned[[i, j]] > random_2d[[i, j]]
    });

    spikes
}

// 测试模块
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_img_to_spike_train() {
        // 创建一个全零的图像数组，长度为 784（模拟 MNIST 数据）
        let image = Array::from_vec(vec![0; 784]);
        let dt = 0.1; // 时间步长 0.1ms
        let training_steps = 3500; // 3500 个时间步
        let input_intensity = 2.0; // 输入强度
        let mut rng = rand::thread_rng();

        // 调用函数生成脉冲序列
        let spikes = img_to_spike_train(&image, dt, training_steps, input_intensity, &mut rng);

        // 验证输出形状
        assert_eq!(spikes.shape(), &[training_steps, 784]);
    }
}
