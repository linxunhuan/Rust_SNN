use pds_snn::builders::DynSnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

fn main() -> io::Result<()> {
    // 网络参数
    let n_inputs = 784;  // 输入神经元数量，对应 MNIST 图像的 28x28 像素
    let n_outputs = 400; // 输出神经元数量
    let n_steps = 70;   // 更新时间步数，与 Python 的 computationSteps 一致

    // 创建 400 个 LifNeuron，调整参数
    let neurons: Vec<LifNeuron> = (0..n_outputs)
        .map(|_| LifNeuron::new(0.2, 0.0, 0.0, 0.3, 0.1)) // 权重 0.2，阈值 0.3
        .collect();

    // 示例权重：每个输出神经元与 784 个输入的连接权重设为 0.2，层内权重设为 0.0
    let weights = vec![vec![0.2; n_inputs]; n_outputs]; // 400 x 784
    let intra_weights = vec![vec![0.0; n_outputs]; n_outputs]; // 400 x 400

    // 构建动态 SNN
    let mut snn = DynSnnBuilder::new(n_inputs)
        .add_layer(neurons, weights, intra_weights)
        .build();

    // 读取 inputSpikes.txt
    let spikes = load_spikes("inputSpikes.txt", n_steps, n_inputs)?;

    // 转置 spikes 以适应 DynSNN 的 process 方法（每个神经元的时间序列）
    let input_spikes: Vec<Vec<u8>> = (0..n_inputs)
        .map(|i| (0..n_steps).map(|t| spikes[t][i]).collect())
        .collect();

    println!("Input spikes: {:?}", input_spikes);

    // 处理脉冲
    let output_spikes = snn.process(&input_spikes);

    println!("Output spikes: {:?}", output_spikes);

    // 计算每个输出神经元的脉冲计数
    let output_counters: Vec<i32> = output_spikes
        .iter()
        .map(|neuron_spikes| neuron_spikes.iter().filter(|&&s| s == 1).count() as i32)
        .collect();

    println!("Output counters: {:?}", output_counters);

    // 写入 outputCounters.txt
    write_counters("outputCounters.txt", &output_counters)?;

    Ok(())
}

fn load_spikes(path: &str, n_steps: usize, n_inputs: usize) -> io::Result<Vec<Vec<u8>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut spikes = Vec::with_capacity(n_steps);

    for (i, line) in reader.lines().enumerate().take(n_steps) {
        let line = line?;
        let row: Vec<u8> = line.chars().map(|c| {
            if c == '0' || c == '1' {
                c.to_digit(10).unwrap() as u8
            } else {
                panic!("Invalid character in line {}: {}", i + 1, c)
            }
        }).collect();
        if row.len() != n_inputs {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Line {}: Expected {} inputs, got {}", i + 1, n_inputs, row.len()),
            ));
        }
        spikes.push(row);
    }
    println!("Loaded {} lines successfully", spikes.len());
    Ok(spikes)
}

// 将脉冲计数写入文件
fn write_counters(path: &str, counters: &[i32]) -> io::Result<()> {
    let mut file = File::create(path)?;
    println!("Writing counters: {:?}", counters);
    for count in counters {
        writeln!(file, "{}", count)?;
    }
    Ok(())
}