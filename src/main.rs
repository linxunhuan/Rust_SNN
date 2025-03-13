use colored::Colorize;
use pds_snn::builders::DynSnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use ndarray::{Array1, Array2, Axis};
use pds_snn::simulation::inputInterface::img_to_spike_train;

use pds_snn::simulation::mnist::load_dataset;
use rand::thread_rng;

// 模拟参数（从 runSimulation.py 移植）
const DT: f64 = 0.1; // 时间步长（毫秒）
const TRAIN_DURATION: usize = 350; // 脉冲序列持续时间（毫秒）
const COMPUTATION_STEPS: usize = (TRAIN_DURATION as f64 / DT) as usize; // 计算步数
const INPUT_INTENSITY: f64 = 2.0; // 输入像素值的归一化强度
const UPDATE_INTERVAL: usize = 100; // 每隔多少张图像评估一次准确率
const N_NEURONS: [usize; 1] = [400]; // 每层神经元数量
const N_INPUTS: usize = 784; // 输入维度（28x28）
const NUMBER_OF_CYCLES: usize = 301; // 模拟的图像数量

fn main() {
    let n_neurons = N_NEURONS[0]; // 神经元数量
    let n_inputs = N_INPUTS; // 输入数量
    let n_instants = COMPUTATION_STEPS; // 计算步数

    // 初始化随机数生成器
    let mut rng = thread_rng();

    // 加载 MNIST 数据集
    let (img_array, labels_array) = load_dataset(
        "/home/luoyin/Rust-SNN-test/src/simulation/mnist/t10k-images-idx3-ubyte",
        "/home/luoyin/Rust-SNN-test/src/simulation/mnist/t10k-labels-idx1-ubyte",
    );

    // 从文件中加载神经元分配
    let assignments: Array1<f64> = ndarray_npy::read_npy(
        "/home/luoyin/Rust-SNN-test/src/simulation/networkParameters/assignments.npy",
    )
    .expect("无法读取 assignments");

    // 初始化脉冲历史
    let mut counters_evolution = Array2::zeros((UPDATE_INTERVAL, n_neurons)); // 计数器演变

    // 输入和输出文件名
    let input_spikes_filename = "/home/luoyin/Rust-SNN-test/src/simulation/inputSpikes.txt"; // 输入脉冲文件
    let output_counters_filename = "/home/luoyin/Rust-SNN-test/src/simulation/outputCounters.txt"; // 输出计数器文件

    println!("{}", "开始模拟...".yellow());

    // 模拟循环
    for i in 0..NUMBER_OF_CYCLES {
        print!("\n迭代: {}\n", i + 1);

        // 将图像转换为脉冲序列
        let image = img_array.index_axis(Axis(0), i).to_owned();
        let spikes_trains = img_to_spike_train(
            &image.mapv(|x| x as u8),
            DT,
            COMPUTATION_STEPS,
            INPUT_INTENSITY,
            &mut rng,
        );

        // 写入输入脉冲文件
        {
            let mut fp = File::create(input_spikes_filename).expect("无法创建输入脉冲文件");
            for step in spikes_trains.outer_iter() {
                let line: String = step.iter().map(|&x| if x { '1' } else { '0' }).collect();
                writeln!(fp, "{}", line).expect("无法写入输入脉冲文件");
            }
        }

        // 构建和处理 SNN
        let building_start = Instant::now();
        let neurons: Vec<LifNeuron> = build_neurons(n_neurons); // 构建神经元
        let extra_weights: Vec<Vec<f64>> = read_extra_weights(n_neurons, n_inputs); // 读取外部权重
        let intra_weights: Vec<Vec<f64>> = build_intra_weights(n_neurons); // 构建内部权重
        let mut snn = DynSnnBuilder::new(n_inputs)
            .add_layer(neurons, extra_weights, intra_weights)
            .build();
        let building_end = building_start.elapsed();
        println!("{}", "SNN 构建完成！".green());
        println!(
            "{}",
            format!(
                "\n构建网络耗时: {}.{:03} 秒\n",
                building_end.as_secs(),
                building_end.subsec_millis()
            )
            .blue()
        );

        let computing_start = Instant::now();
        let input_spikes: Vec<Vec<u8>> = read_input_spikes(n_instants, n_inputs); // 读取输入脉冲
        let output_spikes = snn.process(&input_spikes); // 处理 SNN
        let computing_end = computing_start.elapsed();
        println!("{}", "输出脉冲计算完成！".green());
        println!(
            "{}",
            format!(
                "\n计算输出脉冲耗时: {}.{:03} 秒\n",
                computing_end.as_secs(),
                computing_end.subsec_millis()
            )
            .blue()
        );

        // 写入输出脉冲文件
        write_to_output_file(output_spikes, n_neurons, n_instants);

        // 读取输出计数器
        let mut output_counters = Array1::zeros(n_neurons);
        {
            let fp = File::open(output_counters_filename).expect("无法打开输出计数器文件");
            let buffered = BufReader::new(fp);
            for (j, line) in buffered.lines().enumerate() {
                output_counters[j] = line.unwrap().parse::<i32>().expect("无法解析输出计数器");
            }
        }

        // 更新计数器历史
        counters_evolution
            .row_mut((i % UPDATE_INTERVAL) as usize)
            .assign(&output_counters.mapv(|x| x as i32));

        // 预测标签
        let mut max_count = 0;
        let mut predicted_label = -1;
        for label in 0..10 {
            let spikes_count = output_counters
                .iter()
                .enumerate()
                .filter(|&(idx, _)| assignments[idx] == label as f64)
                .map(|(_, &val)| val)
                .sum::<i32>();
            if spikes_count > max_count {
                max_count = spikes_count;
                predicted_label = label as i32;
            }
        }

        // 真实标签
        let true_label = labels_array[i];

        // 输出预测结果
        println!("预测标签: {}, 真实标签: {}", predicted_label, true_label);
    }

    println!("{}", "模拟完成！".green());
}

// 构建神经元
fn build_neurons(n_neurons: usize) -> Vec<LifNeuron> {
    let thresholds: Vec<f64> = read_thresholds(n_neurons); // 读取阈值
    let v_rest: f64 = -65.0; // 静息电位
    let v_reset: f64 = -60.0; // 重置电位
    let tau: f64 = 100.0; // 时间常数
    let dt: f64 = 0.1; // 时间步长

    let mut neurons: Vec<LifNeuron> = Vec::with_capacity(n_neurons);
    println!("{}", "正在构建神经元...".yellow());
    for i in 0..n_neurons {
        let neuron = LifNeuron::new(thresholds[i], v_rest, v_reset, tau, dt);
        neurons.push(neuron);
    }
    println!("{}", "完成！".green());
    neurons
}

// 构建内部权重
fn build_intra_weights(n_neurons: usize) -> Vec<Vec<f64>> {
    let value: f64 = -15.0; // 内部权重值
    let mut intra_weights: Vec<Vec<f64>> = vec![vec![0f64; n_neurons]; n_neurons];
    println!("{}", "正在构建内部权重...".yellow());
    for i in 0..n_neurons {
        for j in 0..n_neurons {
            if i == j {
                intra_weights[i][j] = 0.0; // 自连接为 0
            } else {
                intra_weights[i][j] = value; // 其他连接为负值
            }
        }
    }
    println!("{}", "完成！".green());
    intra_weights
}

// 读取输入脉冲
fn read_input_spikes(n_instants: usize, n_inputs: usize) -> Vec<Vec<u8>> {
    let path_input = "/home/luoyin/Rust-SNN-test/src/simulation/inputSpikes.txt";
    let input = File::open(path_input).expect("无法打开 inputSpikes.txt 文件！");
    let buffered = BufReader::new(input);
    let mut input_spikes: Vec<Vec<u8>> = vec![vec![0; n_instants]; n_inputs];
    println!("{}", "正在从 inputSpikes.txt 读取输入脉冲...".yellow());
    let mut i = 0;
    for line in buffered.lines() {
        let chars = convert_line_into_u8(line.unwrap());
        for (j, ch) in chars.into_iter().enumerate() {
            input_spikes[j][i] = ch;
        }
        i += 1;
    }
    println!("{}", "完成！".green());
    input_spikes
}

// 读取外部权重
fn read_extra_weights(n_neurons: usize, n_inputs: usize) -> Vec<Vec<f64>> {
    let path_weights_file =
        "/home/luoyin/Rust-SNN-test/src/simulation/networkParameters/weightsOut.txt";
    let input = File::open(path_weights_file).expect("无法打开 weightsOut.txt 文件！");
    let buffered = BufReader::new(input);
    let mut extra_weights: Vec<Vec<f64>> = vec![vec![0f64; n_inputs]; n_neurons];
    println!("{}", "正在从 weightsOut.txt 读取权重...".yellow());
    let mut i = 0;
    for line in buffered.lines() {
        let split: Vec<String> = line.unwrap().split(" ").map(|el| el.to_string()).collect();
        for j in 0..n_inputs {
            extra_weights[i][j] = split[j].parse::<f64>().expect("无法将字符串解析为 f64！");
        }
        i += 1;
    }
    println!("{}", "完成！".green());
    extra_weights
}

// 读取阈值
fn read_thresholds(n_neurons: usize) -> Vec<f64> {
    let path_threshold_file =
        "/home/luoyin/Rust-SNN-test/src/simulation/networkParameters/thresholdsOut.txt";
    let input = File::open(path_threshold_file).expect("无法打开 thresholdsOut.txt 文件！");
    let buffered = BufReader::new(input);
    let mut thresholds: Vec<f64> = vec![0f64; n_neurons];
    println!("{}", "正在从 thresholdsOut.txt 读取阈值...".yellow());
    let mut i = 0;
    for line in buffered.lines() {
        thresholds[i] = line
            .unwrap()
            .parse::<f64>()
            .expect("无法将字符串解析为 f64！");
        i += 1;
    }
    println!("{}", "完成！".green());
    thresholds
}

// 写入输出文件
fn write_to_output_file(output_spikes: Vec<Vec<u8>>, n_neurons: usize, n_instants: usize) {
    let path_output = "/home/luoyin/Rust-SNN-test/src/simulation/outputCounters.txt";
    let mut output_file = File::create(path_output).expect("无法打开 outputCounters.txt 文件！");
    let mut neurons_sum: Vec<u32> = vec![0; n_neurons];
    println!("{}", "正在计算每个神经元的脉冲总和...".yellow());
    for i in 0..n_neurons {
        for j in 0..n_instants {
            neurons_sum[i] += output_spikes[i][j] as u32;
        }
    }
    println!("{}", "完成！".green());
    println!("{}", "正在将数据写入 outputCounters.txt 文件...".yellow());
    for i in 0..n_neurons {
        output_file
            .write_all(format!("{}\n", neurons_sum[i]).as_bytes())
            .expect("无法写入 outputCounters.txt 文件！");
    }
    println!("{}", "完成！".green());
}

// 将字符串转换为 u8 向量
fn convert_line_into_u8(line: String) -> Vec<u8> {
    line.chars()
        .map(|ch| ch.to_digit(10).unwrap() as u8)
        .collect::<Vec<u8>>()
}
