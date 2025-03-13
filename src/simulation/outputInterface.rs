use ndarray::{Array2, Array1, Axis};
use ndarray::s; // Import the s macro
use ndarray::ArrayBase; // Import ArrayBase
use ndarray::Data; // Import Data trait
use ndarray::Dim; // Import Dim
use std::fmt::Write;

/// 根据脉冲计数器计算网络性能并更新准确率。
///
/// # 输入参数
/// - `current_index`: 当前图像的索引 (`usize`)。
/// - `update_interval`: 计算性能的图像间隔数 (`usize`)。
/// - `counters_evolution`: 最近 `update_interval` 周期内脉冲计数器的二维数组。
///   形状: `(update_interval, n_neurons)`，类型: `&Array2<i32>`。
/// - `labels`: 所有图像标签的一维数组，类型: `&Array1<i32>`。
/// - `assignments`: 每个输出神经元的标签分配一维数组，类型: `&Array1<i32>`。
/// - `accuracies`: 包含准确率历史的字符串向量可变引用 (`&mut Vec<String>`)。
///
/// # 输出
/// 更新后的 `accuracies` 向量，附加了新的准确率。
///
/// # 描述
/// 该函数每隔 `update_interval` 个图像通过分析输出层的脉冲计数器来评估网络性能。
/// 它基于脉冲计数最高的神经元对图像进行分类，并更新准确率历史记录。
pub fn compute_performance<'a>(
    current_index: usize,
    update_interval: usize,
    counters_evolution: &'a Array2<i32>,
    labels: &'a Array1<i32>,
    assignments: &'a Array1<i32>,
    accuracies: &'a mut Vec<String>,
) -> &'a mut Vec<String> {
    // 检查是否到了计算性能的时间
    if current_index % update_interval == 0 && current_index > 0 {
        // 使用全零初始化 max_count
        let mut max_count = Array1::<f64>::zeros(update_interval);
        // 使用 -1 初始化 classification
        let mut classification = Array1::<i32>::from_elem(update_interval, -1);

        // 提取最近 update_interval 个图像的标签序列
        let labels_sequence = labels.slice(s![current_index - update_interval..current_index]);

        // 遍历每个可能的标签 (0 到 9)
        for label in 0..10 {
            // 创建一个掩码，where assignments 等于当前标签
            let mask = assignments.mapv(|a| a == label);
            // 选择具有此标签的神经元并沿轴 1 求和它们的脉冲计数
            let indices: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter(|&(_, &b)| b)
                .map(|(i, _)| i)
                .collect();
            let spikes_count = counters_evolution
                .select(Axis(1), &indices)
                .sum_axis(Axis(1))
                .mapv(|x| x as f64);

            // 查找 spikes_count 超过 max_count 的位置 (element-wise comparison)
            let where_max_spikes = Array1::from_shape_fn(update_interval, |i| {
                spikes_count[i] > max_count[i]
            });

            // 在 spikes_count 较大时更新 classification 和 max_count
            for (i, &cond) in where_max_spikes.iter().enumerate() {
                if cond {
                    classification[i] = label;
                    max_count[i] = spikes_count[i];
                }
            }
        }

        // 打印计算出的分类结果
        println!("{:?}", classification);

        // 更新准确率历史
        update_accuracy(&classification, &labels_sequence, accuracies);
    }

    accuracies
}

/// 根据分类结果更新准确率历史。
///
/// # 输入参数
/// - `classification`: 最近 `update_interval` 周期内网络分类结果的一维数组，
///   类型: `&Array1<i32>`。
/// - `labels_sequence`: 最近 `update_interval` 周期内真实标签的一维数组，
///   类型: `&ArrayBase<impl Data<Elem=i32>, Dim<[usize; 1]>>`。
/// - `accuracies`: 包含准确率历史的字符串向量可变引用 (`&mut Vec<String>`)。
///
/// # 输出
/// 更新后的 `accuracies` 向量，附加了新的准确率。
///
/// # 描述
/// 该函数计算正确分类的百分比，并将其作为格式化字符串附加到准确率历史中。
fn update_accuracy<'a>(
    classification: &'a Array1<i32>,
    labels_sequence: &'a ArrayBase<impl Data<Elem=i32>, Dim<[usize; 1]>>, // Accept view type
    accuracies: &'a mut Vec<String>,
) -> &'a mut Vec<String> {
    // 统计正确分类的个数
    let correct = classification
        .iter()
        .zip(labels_sequence.iter())
        .filter(|(&c, &l)| c == l)
        .count();

    // 计算准确率百分比
    let accuracy = (correct as f64 / classification.len() as f64) * 100.0;

    // 将格式化的准确率附加到列表
    accuracies.push(format!("{:.2}%", accuracy));

    // 打印准确率历史
    let mut accuracy_string = String::new();
    write!(&mut accuracy_string, "\nAccuracy: {:?}\n", accuracies).unwrap();
    print!("{}", accuracy_string);

    accuracies
}

// 测试模块
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_performance() {
        // 测试数据
        let update_interval = 5;
        let current_index = 5; // 恰好达到 update_interval
        let n_neurons = 10;

        // 模拟 counters_evolution: 5 个时间步，10 个神经元
        let counters_evolution = Array2::from_shape_vec(
            (update_interval, n_neurons),
            vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 时间步 0
                0, 2, 0, 0, 0, 0, 0, 0, 0, 0, // 时间步 1
                0, 0, 3, 0, 0, 0, 0, 0, 0, 0, // 时间步 2
                0, 0, 0, 4, 0, 0, 0, 0, 0, 0, // 时间步 3
                0, 0, 0, 0, 5, 0, 0, 0, 0, 0, // 时间步 4
            ],
        )
        .unwrap();

        // 模拟 labels: 前 5 个标签
        let labels = Array1::from_vec(vec![0, 1, 2, 3, 4]);

        // 模拟 assignments: 每个神经元分配给对应的标签 (0 到 9)
        let assignments = Array1::from_vec((0..10).collect());

        // 初始化 accuracies
        let mut accuracies = Vec::new();

        // 调用 compute_performance (ignore unused result)
        let _result = compute_performance(current_index, update_interval, &counters_evolution, &labels, &assignments, &mut accuracies);

        // 验证结果 (use accuracies directly after mutable borrow ends)
        assert_eq!(accuracies.len(), 1); // 应该添加一个准确率
        assert!(accuracies[0].ends_with("%")); // 确保格式正确
        assert_eq!(accuracies[0], "100.00%"); // 由于每个时间步的脉冲计数最大值对应正确标签，准确率应为 100%

        // 验证 classification (预期应为 [0, 1, 2, 3, 4])
        let mut classification = Array1::<i32>::from_elem(update_interval, -1);
        let mut max_count = Array1::<f64>::zeros(update_interval);
        for label in 0..10 {
            let mask = assignments.mapv(|a| a == label);
            let indices: Vec<usize> = mask.iter().enumerate().filter(|&(_, &b)| b).map(|(i, _)| i).collect();
            let spikes_count = counters_evolution.select(Axis(1), &indices).sum_axis(Axis(1)).mapv(|x| x as f64);
            let where_max_spikes = Array1::from_shape_fn(update_interval, |i| {
                spikes_count[i] > max_count[i]
            });
            for (i, &cond) in where_max_spikes.iter().enumerate() {
                if cond {
                    classification[i] = label;
                    max_count[i] = spikes_count[i];
                }
            }
        }
        let expected_classification = Array1::from_vec(vec![0, 1, 2, 3, 4]);
        assert_eq!(classification, expected_classification);
    }

    #[test]
    fn test_update_accuracy() {
        // 测试数据
        let classification = Array1::from_vec(vec![0, 1, 2, 3, 4]); // 正确分类
        let labels_sequence = Array1::from_vec(vec![0, 1, 2, 3, 4]); // 真实标签
        let mut accuracies = Vec::new();

        // 调用 update_accuracy
        let _result = update_accuracy(&classification, &labels_sequence, &mut accuracies);

        // 验证结果
        assert_eq!(accuracies.len(), 1); // 应该添加一个准确率
        assert_eq!(accuracies[0], "100.00%"); // 所有分类正确，准确率应为 100%

        // 测试部分正确的情况
        let classification_partial = Array1::from_vec(vec![0, 1, 2, 3, 5]); // 最后一个错误
        let labels_sequence_partial = Array1::from_vec(vec![0, 1, 2, 3, 4]);
        let mut accuracies_partial = Vec::new();
        let _result_partial = update_accuracy(&classification_partial, &labels_sequence_partial, &mut accuracies_partial);

        assert_eq!(accuracies_partial.len(), 1);
        assert_eq!(accuracies_partial[0], "80.00%"); // 4/5 正确，准确率应为 80%
    }
}