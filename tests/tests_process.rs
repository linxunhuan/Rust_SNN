use pds_snn::builders::{DynSnnBuilder, SnnBuilder};
use pds_snn::models::neuron::lif::LifNeuron;

// 打印从SNN处理获得的输出脉冲的函数
fn print_output(test_name: &str, output_spikes: Vec<Vec<u8>>) -> () {
    // 打印输出脉冲的标题
    println!("\nOUTPUT SPIKES for {}:\n", test_name);
    print!("t   "); // 打印时间轴标签

    // 遍历输出脉冲，并为每个脉冲序列打印时间轴和脉冲值
    for (n, spikes) in output_spikes.into_iter().enumerate() {
        if n == 0 {
            // 如果是第一个脉冲序列，则打印时间轴
            (0..spikes.len()).for_each(|t| print!("{} ", t));
            println!();
        }

        // 打印神经元编号
        print!("N{}  ", n);

        // 打印每个时间点的脉冲值
        for spike in spikes {
            print!("{} ", spike);
        }
        println!(); // 打印新行
    }
    println!(); // 打印新行以便格式更整齐
}

// 与SNN处理函数相关的测试

#[test]
fn test_process_snn_with_only_one_layer() {
    #[rustfmt::skip]
    
    // 创建并构建只有一层的SNN
    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],         // 第一层神经元的权重
            [0.3, 0.4],
            [0.5, 0.6]
        ]).neurons([
            // 3个LIF神经元
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        ]).intra_weights([
            [0.0, -0.1, -0.15], // 第一层神经元的内部权重
            [-0.05, 0.0, -0.1],
            [-0.15, -0.1, 0.0]
        ]).build();

    // 处理输入脉冲
    let output_spikes = snn.process(&[[1, 0, 1], [0, 0, 1]]);
    let output_expected: [[u8; 3]; 3] = [[0, 0, 0], [1, 0, 1], [1, 0, 1]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());
}


// 测试与动态SNN处理函数相关的功能

#[test]
fn test_process_dyn_snn_with_only_one_layer() {
    #[rustfmt::skip]
    
    // 创建并构建只有一层的动态SNN
    let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            // 3个LIF神经元
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
        ], vec![
            vec![0.1, 0.2],          // 第一层神经元的权重
            vec![0.3, 0.4],
            vec![0.5, 0.6]
        ], vec![
            vec![0.0, -0.1, -0.15],  // 第一层神经元的内部权重
            vec![-0.05, 0.0, -0.1],
            vec![-0.15, -0.1, 0.0]
        ])
        .build();

    // 处理输入脉冲
    let output_spikes = snn.process(&vec![vec![1, 0, 1], vec![0, 0, 1]]);
    let output_expected: Vec<Vec<u8>> = vec![vec![0, 0, 0], vec![1, 0, 1], vec![1, 0, 1]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_dyn_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());
}


// 测试与SNN处理函数相关的功能，使用不同的时间步长（dt）

#[test]
fn test_process_snn_with_only_one_layer_and_different_dt() {
    #[rustfmt::skip]
    
    let dt = 0.1;

    // 创建并构建只有一层的SNN，使用不同的时间步长（dt）
    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],         // 第一层神经元的权重
            [0.3, 0.4],
            [0.5, 0.6]
        ]).neurons([
            // 3个LIF神经元，使用不同的时间步长（dt）
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
        ]).intra_weights([
            [0.0, -0.1, -0.15], // 第一层神经元的内部权重
            [-0.05, 0.0, -0.1],
            [-0.15, -0.1, 0.0]
        ]).build();

    // 处理输入脉冲
    let output_spikes = snn.process(&[[1, 0, 1], [0, 0, 1]]);
    let output_expected: [[u8; 3]; 3] = [[0, 0, 0], [1, 0, 1], [1, 0, 1]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());
}

// 测试与动态SNN处理函数相关的功能，使用不同的时间步长（dt）
#[test]
fn test_process_dyn_snn_with_only_one_layer_and_different_dt() {
    #[rustfmt::skip]
    
    let dt = 0.1;

    // 创建并构建只有一层且使用不同时间步长的动态SNN
    let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            // 3个LIF神经元，使用不同的时间步长（dt）
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt)
        ], vec![
            vec![0.1, 0.2],          // 第一层神经元的权重
            vec![0.3, 0.4],
            vec![0.5, 0.6]
        ], vec![
            vec![0.0, -0.1, -0.15],  // 第一层神经元的内部权重
            vec![-0.05, 0.0, -0.1],
            vec![-0.15, -0.1, 0.0]
        ])
        .build();

    // 处理输入脉冲
    let output_spikes = snn.process(&vec![vec![1, 0, 1], vec![0, 0, 1]]);
    let output_expected: Vec<Vec<u8>> = vec![vec![0, 0, 0], vec![1, 0, 1], vec![1, 0, 1]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_dyn_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());
}


// 测试与SNN处理函数相关的功能，具有多个层且所有神经元相同

#[test]
fn test_process_snn_with_more_than_one_layer_and_same_neurons() {
    #[rustfmt::skip]

    // 创建并构建具有多个层的SNN
    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],          // 第一层神经元的权重
            [0.3, 0.4],
            [0.5, 0.6]
        ]).neurons([
            // 3个LIF神经元
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        ]).intra_weights([
            [0.0, -0.1, -0.15], // 第一层神经元的内部权重
            [-0.05, 0.0, -0.1],
            [-0.15, -0.1, 0.0]
        ]).add_layer()
        .weights([
            [0.3, 0.2, 0.1]      // 第二层神经元的权重
        ]).neurons([
            // 1个LIF神经元
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        ]).intra_weights([[0.0]])
        .add_layer()
        .weights([
            [0.3],               // 第三层神经元的权重
            [0.2],
            [0.5],
            [0.3]
        ]).neurons([
            // 4个LIF神经元
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
        ]).intra_weights([
            [0.0, -0.1, -0.2, -0.3],  // 第三层神经元的内部权重
            [-0.1, 0.0, -0.4, -0.2],
            [-0.6, -0.2, 0.0, -0.9],
            [-0.5, -0.3, -0.8, 0.0]
        ]).build();

    // 处理输入脉冲
    let output_spikes = snn.process(&[[1, 0, 1], [0, 0, 1]]);
    let output_expected: [[u8; 3]; 4] = [[1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_snn_with_more_than_one_layer_and_same_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());
}


// 测试与SNN处理函数相关的功能，具有一层且神经元不同
#[test]
fn test_process_snn_with_only_one_layer_and_different_neurons() {
    #[rustfmt::skip]
    
    // 创建并构建只有一层且神经元不同的SNN
    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3, 0.4],   // 第一层神经元的权重
            [0.1, 0.4, 0.1, 0.2],
            [0.5, 0.1, 0.7, 0.25]
        ]).neurons([
            // 3个不同参数的LIF神经元
            LifNeuron::new(0.31, 0.01, 0.1, 0.8, 1.0),
            LifNeuron::new(0.32, 0.02, 0.3, 0.9, 1.0),
            LifNeuron::new(0.33, 0.03, 0.2, 1.0, 1.0),
        ]).intra_weights([
            [0.0, -0.6, -0.3],     // 第一层神经元的内部权重
            [-0.5, 0.0, -0.15],
            [-0.4, -0.05, 0.0]
        ]).build();

    // 处理输入脉冲
    let output_spikes = snn.process(&[[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]);
    let output_expected: [[u8; 3]; 3] = [[0, 1, 0], [0, 1, 0], [1, 1, 1]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_snn_with_only_one_layer_and_different_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());
}


// 测试与SNN处理函数相关的功能，具有多个层且神经元不同
#[test]
fn test_process_snn_with_more_than_one_layer_and_different_neurons() {
    #[rustfmt::skip]
    
    // 创建并构建具有多个层且神经元不同的SNN
    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],          // 第一层神经元的权重
            [0.3, 0.4]
        ]).neurons([
            // 2个不同参数的LIF神经元
            LifNeuron::new(0.5, 0.1, 0.2, 0.7, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        ]).intra_weights([
            [0.0, -0.4],         // 第一层神经元的内部权重
            [-0.1, 0.0]
        ]).add_layer()
        .weights([
            [0.7, 0.2],          // 第二层神经元的权重
            [0.3, 0.8],
            [0.5, 0.6],
            [0.3, 0.2]
        ]).neurons([
            // 4个不同参数的LIF神经元
            LifNeuron::new(0.2, 0.1, 0.15, 0.1, 1.0),
            LifNeuron::new(0.3, 0.2, 0.05, 0.3, 1.0),
            LifNeuron::new(0.4, 0.15, 0.1, 0.8, 1.0),
            LifNeuron::new(0.05, 0.35, 0.01, 1.0, 1.0),
        ]).intra_weights([
            [0.0, -0.2, -0.4, -0.9],  // 第二层神经元的内部权重
            [-0.1, 0.0, -0.3, -0.2],
            [-0.6, -0.2, 0.0, -0.9],
            [-0.5, -0.3, -0.8, 0.0]
        ])
        .add_layer()
        .weights([
            [0.3, 0.3, 0.2, 0.7]    // 第三层神经元的权重
        ]).neurons([
            // 1个LIF神经元
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
        ]).intra_weights([
            [0.0]
        ]).build();

    // 处理输入脉冲
    let output_spikes = snn.process(&[[1, 0, 1, 0], [0, 0, 1, 1]]);
    let output_expected: [[u8; 4]; 1] = [[1, 0, 1, 1]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_snn_with_more_than_one_layer_and_different_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());
}


// 测试与动态SNN处理函数相关的功能，具有多个层且神经元不同
#[test]
fn test_process_dyn_snn_with_more_than_one_layer_and_different_neurons() {
    #[rustfmt::skip]
    
    // 创建并构建具有多个层且神经元不同的动态SNN
    let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            // 第一层中的2个不同参数的LIF神经元
            LifNeuron::new(0.5, 0.1, 0.2, 0.7, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
        ], vec![
            vec![0.1, 0.2],          // 第一层神经元的权重
            vec![0.3, 0.4]
        ], vec![
            vec![0.0, -0.4],         // 第一层神经元的内部权重
            vec![-0.1, 0.0]
        ])
        .add_layer(vec![
            // 第二层中的4个不同参数的LIF神经元
            LifNeuron::new(0.2, 0.1, 0.15, 0.1, 1.0),
            LifNeuron::new(0.3, 0.2, 0.05, 0.3, 1.0),
            LifNeuron::new(0.4, 0.15, 0.1, 0.8, 1.0),
            LifNeuron::new(0.05, 0.35, 0.01, 1.0, 1.0)
        ], vec![
            vec![0.7, 0.2],          // 第二层神经元的权重
            vec![0.3, 0.8],
            vec![0.5, 0.6],
            vec![0.3, 0.2]
        ], vec![
            vec![0.0, -0.2, -0.4, -0.9],  // 第二层神经元的内部权重
            vec![-0.1, 0.0, -0.3, -0.2],
            vec![-0.6, -0.2, 0.0, -0.9],
            vec![-0.5, -0.3, -0.8, 0.0]
        ])
        .add_layer(vec![
            // 第三层中的1个LIF神经元
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
        ], vec![
            vec![0.3, 0.3, 0.2, 0.7]   // 第三层神经元的权重
        ], vec![
            vec![0.0]
        ])
        .build();

    // 处理输入脉冲
    let output_spikes = snn.process(&vec![vec![1, 0, 1, 0], vec![0, 0, 1, 1]]);
    let output_expected: Vec<Vec<u8>> = vec![vec![1, 0, 1, 1]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_dyn_snn_with_more_than_one_layer_and_different_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());
}


// 测试与SNN处理函数相关的功能，具有不同的神经元且输入脉冲超过五个
#[test]
fn test_process_snn_with_different_neurons_and_more_than_five_input_spikes() {
    #[rustfmt::skip]

    // 创建并构建具有不同神经元且输入脉冲超过五个的SNN
    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.3, 0.21, 0.36, 0.47],   // 第一层神经元的权重
            [0.6, 0.45, 0.34, 0.21],
            [0.1, 0.62, 0.72, 0.82],
            [0.12, 0.23, 0.6, 0.8]
        ]).neurons([
            // 4个不同参数的LIF神经元
            LifNeuron::new(0.67, 0.01, 0.1, 0.8, 1.0),
            LifNeuron::new(0.4, 0.02, 0.3, 0.9, 1.0),
            LifNeuron::new(0.33, 0.03, 0.2, 1.0, 1.0),
            LifNeuron::new(0.9, 0.05, 0.7, 0.5, 1.0),
        ]).intra_weights([
            [0.0, -0.6, -0.3, -0.2],  // 第一层神经元的内部权重
            [-0.5, 0.0, -0.15, -0.4],
            [-0.4, -0.05, 0.0, -0.2],
            [-0.1, -0.25, -0.15, 0.0]
        ]).build();

    // 处理输入脉冲
    let output_spikes = snn.process(&[[1, 1, 0, 1, 0, 1, 1, 1, 1], [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0]]);
    let output_expected: [[u8; 9]; 4] = [[0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_snn_with_different_neurons_and_more_than_five_input_spikes", output_spikes.iter().map(|x| x.to_vec()).collect());
}


// 测试与SNN处理函数相关的功能，输入全为零
#[test]
fn test_process_snn_with_all_zeros_as_input() {
    #[rustfmt::skip]
    
    // 创建并构建只有一层且输入全为零的SNN
    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3],   // 第一层神经元的权重
            [0.1, 0.4, 0.3],
            [0.5, 0.6, 0.7]
        ]).neurons([
            // 3个不同参数的LIF神经元
            LifNeuron::new(0.31, 0.01, 0.1, 0.8, 1.0),
            LifNeuron::new(0.32, 0.02, 0.3, 0.9, 1.0),
            LifNeuron::new(0.33, 0.03, 0.2, 1.0, 1.0),
        ]).intra_weights([
            [0.0, -0.6, -0.3], // 第一层神经元的内部权重
            [-0.5, 0.0, -0.15],
            [-0.4, -0.05, 0.0]
        ]).build();

    // 处理输入脉冲，全为零
    let output_spikes = snn.process(&[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
    let output_expected: [[u8; 4]; 3] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];

    // 断言输出结果是否与预期一致
    assert_eq!(output_spikes, output_expected);

    // 打印输出脉冲
    print_output("test_process_snn_with_all_zeros_as_input", output_spikes.iter().map(|x| x.to_vec()).collect());
}


#[test]  
fn test_process_snn_with_zero_inputs() { 
    #[rustfmt::skip]  // 禁用 rustfmt 格式化器的自动格式化，保持代码原样

    // 构建一个 SNN 网络
    let mut snn = SnnBuilder::new()  
        .add_layer()  // 添加一层神经元
        .weights([  // 设置层与层之间的权重矩阵 (输入到神经元)
            [0.1, 0.2, 0.3],  // 第一行：输入 1 到 3 对应神经元的权重
            [0.1, 0.4, 0.3],  // 第二行：输入 1 到 3 对应神经元的权重
            [0.5, 0.6, 0.7],  // 第三行：输入 1 到 3 对应神经元的权重
        ])
        .neurons([  // 设置神经元的属性 (LIF 神经元模型)
            LifNeuron::new(0.31, 0.01, 0.1, 0.8, 1.0),  // 第一个神经元的参数（膜电位阈值，时间常数等）
            LifNeuron::new(0.32, 0.02, 0.3, 0.9, 1.0),  // 第二个神经元的参数
            LifNeuron::new(0.33, 0.03, 0.2, 1.0, 1.0),  // 第三个神经元的参数
        ])
        .intra_weights([  // 设置神经元内部的连接权重（层内连接）
            [0.0, -0.6, -0.3],  // 神经元 1 与神经元 2、3 的连接权重
            [-0.5, 0.0, -0.15],  // 神经元 2 与神经元 1、3 的连接权重
            [-0.4, -0.05, 0.0],  // 神经元 3 与神经元 1、2 的连接权重
        ])
        .build();  // 完成 SNN 构建

    // 传入一个包含空输入信号的三维向量（模拟没有输入）
    let output_spikes = snn.process(&[[], [], []]);  // 调用 `process` 方法，处理没有输入信号的情形

    // 预期的输出是三个空的脉冲序列（因为没有输入）
    let output_expected: [[u8; 0]; 3] = [[], [], []];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_zero_inputs", output_spikes.iter().map(|x| x.to_vec()).collect());
}


#[test]  
fn test_process_snn_with_only_one_input() {  
    #[rustfmt::skip]  

    let mut snn = SnnBuilder::new()  
        .add_layer()  // 添加一层神经元
        .weights([  // 设置层与层之间的权重矩阵 (输入到神经元)
            [0.1, 0.2],  // 第一行：输入 1 到 2 对应神经元的权重
            [0.3, 0.4],  // 第二行：输入 1 到 2 对应神经元的权重
            [0.5, 0.25],  // 第三行：输入 1 到 2 对应神经元的权重
        ])
        .neurons([  // 设置神经元的属性 (LIF 神经元模型)
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第一个神经元的参数（膜电位阈值，时间常数等）
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第二个神经元的参数
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第三个神经元的参数
        ])
        .intra_weights([  // 设置神经元内部的连接权重（层内连接）
            [0.0, -0.1, -0.15],  // 神经元 1 与神经元 2、3 的连接权重
            [-0.05, 0.0, -0.1],   // 神经元 2 与神经元 1、3 的连接权重
            [-0.15, -0.1, 0.0],   // 神经元 3 与神经元 1、2 的连接权重
        ])
        .build();  // 完成 SNN 构建

    // 传入一个包含两个输入信号的二维向量，第一个输入信号是 [0]，第二个输入信号是 [1]
    // 这模拟了只有一个神经元接收到一个输入信号，另一个接收到零输入
    let output_spikes = snn.process(&[[0], [1]]);  // 调用 `process` 方法，处理输入信号

    // 预期的输出是三个脉冲序列，第一个神经元没有激发脉冲（[0]），第二个神经元激发了脉冲（[1]），
    // 第三个神经元没有激发脉冲（[0]）
    let output_expected: [[u8; 1]; 3] = [[0], [1], [0]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_only_one_input", output_spikes.iter().map(|x| x.to_vec()).collect());
}


#[test]
fn test_process_snn_with_all_zeros_as_extra_weights() {  
    #[rustfmt::skip]  

    let mut snn = SnnBuilder::new()  
        .add_layer()  // 添加一层神经元
        .weights([  // 设置层与层之间的权重矩阵 (输入到神经元)，这里所有外部连接的权重都是零
            [0.0, 0.0, 0.0, 0.0],  // 第一行：输入 1 到 4 对应神经元的权重（都是零）
            [0.0, 0.0, 0.0, 0.0],  // 第二行：输入 1 到 4 对应神经元的权重（都是零）
            [0.0, 0.0, 0.0, 0.0]   // 第三行：输入 1 到 4 对应神经元的权重（都是零）
        ])
        .neurons([  // 设置神经元的属性 (LIF 神经元模型)
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第一个神经元的参数（膜电位阈值，时间常数等）
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第二个神经元的参数
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第三个神经元的参数
        ])
        .intra_weights([  // 设置神经元之间的内部连接权重（层内连接）
            [0.0, -0.5, -0.6],  // 神经元 1 与神经元 2、3 的连接权重
            [-0.3, 0.0, -0.2],   // 神经元 2 与神经元 1、3 的连接权重
            [-0.7, -0.3, 0.0]    // 神经元 3 与神经元 1、2 的连接权重
        ])
        .build();  // 完成 SNN 构建

    // 传入一个包含四个时间步的输入信号，每个时间步都有四个输入信号
    // 这些输入信号模拟了网络的活动状态。
    // 例如，第一个时间步是 [1, 1, 0, 0]，表示神经元 1 和 2 激活，神经元 3 和 4 没有激活。
    let output_spikes = snn.process(&[[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]);

    // 预期的输出是三个脉冲序列，这里的每个脉冲序列应该是空的，因为所有输入权重为零。
    // 由于输入信号全为零，神经元不会触发任何脉冲，所以每个神经元的脉冲序列为空。
    let output_expected: [[u8; 3]; 3] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_all_zeros_as_extra_weights", output_spikes.iter().map(|x| x.to_vec()).collect());
}


#[test]  
fn test_process_snn_with_all_zeros_as_intra_weights() {  
    #[rustfmt::skip] 

    let mut snn = SnnBuilder::new()  
        .add_layer()  // 添加一层神经元
        .weights([  // 设置层与层之间的外部连接权重（输入到神经元）
            [0.4, 0.2, 0.4, 0.6],  // 第一行：输入 1 到 4 对应神经元的权重
            [0.7, 0.3, 0.5, 0.8],  // 第二行：输入 1 到 4 对应神经元的权重
            [0.1, 0.2, 0.7, 0.9],  // 第三行：输入 1 到 4 对应神经元的权重
        ])
        .neurons([  // 设置神经元的属性 (LIF 神经元模型)
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第一个神经元的参数（膜电位阈值，时间常数等）
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第二个神经元的参数
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第三个神经元的参数
        ])
        .intra_weights([  // 设置神经元之间的内部连接权重（层内连接），所有值都为零
            [0.0, 0.0, 0.0],  // 神经元 1 与神经元 2、3 的内部连接权重
            [0.0, 0.0, 0.0],  // 神经元 2 与神经元 1、3 的内部连接权重
            [0.0, 0.0, 0.0],  // 神经元 3 与神经元 1、2 的内部连接权重
        ])
        .build();  // 完成 SNN 构建

    // 传入一个包含四个时间步的输入信号，每个时间步有四个输入信号
    // 每个子数组表示在某个时间步，四个神经元的输入状态（1 表示激活，0 表示没有激活）。
    let output_spikes = snn.process(&[[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]);

    // 预期的输出是三个脉冲序列，根据输入信号和神经元的外部权重，
    // 神经元间的内部连接权重为零，意味着神经元之间不相互作用。
    // 预期的脉冲输出如下：
    let output_expected: [[u8; 3]; 3] = [[1, 1, 1], [1, 1, 1], [0, 1, 1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_all_zeros_as_intra_weights", output_spikes.iter().map(|x| x.to_vec()).collect());
}


#[test]  
#[should_panic]  
fn test_input_spikes_greater_than_one() {  
    #[rustfmt::skip]  

    let mut snn = SnnBuilder::new()  
        .add_layer()  
        .weights([  // 设置层与层之间的连接权重（输入到神经元的外部权重）
            [0.12, 0.5],  // 第一行：输入信号 1 和 2 对应神经元的权重
            [0.53, 0.43]  // 第二行：输入信号 1 和 2 对应神经元的权重
        ])
        .neurons([  // 设置神经元的属性（使用 LIF 神经元模型）
            LifNeuron::new(0.3, 0.05, 0.84, 1.0, 1.0),  // 神经元 1 的参数：膜电位阈值、时间常数、外部刺激等
            LifNeuron::new(0.3, 0.87, 0.12, 0.89, 1.0)  // 神经元 2 的参数
        ])
        .intra_weights([  // 设置神经元之间的内部连接权重（层内连接）
            [0.0, -0.3],  // 神经元 1 与神经元 2 的连接权重
            [-0.4, 0.0]    // 神经元 2 与神经元 1 的连接权重
        ])
        .build();  // 完成 SNN 的构建

    // 传入一个输入信号，包含两个时间步，每个时间步都有两个输入信号
    // 在这个例子中，第一个时间步的输入是 [0, 50]，第二个时间步是 [0, 1]。
    // 注意，50 的脉冲输入大于 1，这是不被允许的，因此期望会触发 panic。
    let _output_spikes = snn.process(&[[0, 50], [0, 1]]);
}


#[test] 
#[should_panic]  
fn test_dyn_snn_input_spikes_greater_than_one() {  
    #[rustfmt::skip] 

    let mut snn = DynSnnBuilder::new(2)  // 创建一个新的动态 SNN 实例，指定层数为 2
        .add_layer(  
            vec![  // 第一层神经元的属性，包含 2 个神经元，每个神经元使用 LifNeuron 模型
                LifNeuron::new(0.3, 0.05, 0.84, 1.0, 1.0),  // 第一个神经元的属性：膜电位阈值、时间常数、外部刺激等
                LifNeuron::new(0.3, 0.87, 0.12, 0.89, 1.0),  // 第二个神经元的属性
            ], 
            vec![  // 第一层神经元的输入权重（外部权重）
                vec![0.12, 0.5],  // 输入信号 1 和 2 对神经元的影响
                vec![0.53, 0.43], // 输入信号 1 和 2 对神经元的影响
            ], 
            vec![  // 第一层神经元的内部连接权重（层内连接）
                vec![0.0, -0.3],  // 神经元 1 与神经元 2 之间的连接权重
                vec![-0.4, 0.0]    // 神经元 2 与神经元 1 之间的连接权重
            ]
        ).build();  // 完成 SNN 网络的构建

    // 传入一个包含两个时间步的输入信号
    // 第一个时间步的输入是 [0, 50]，这意味着第一个神经元没有激活，而第二个神经元激活的输入值为 50。
    // 第二个时间步的输入是 [0, 1]，表示第二个神经元激活，而第一个神经元没有激活。
    // 由于输入值 50 大于 1，这是不合法的，期望触发 panic。
    let _output_spikes = snn.process(&vec![vec![0,50], vec![0,1]]);
}


#[test]  
#[should_panic]  
fn test_dyn_snn_input_spikes_greater_than_input_dimension() {  
    #[rustfmt::skip]  
    /*
        该测试用例需要触发 panic，因为输入脉冲的维度与神经网络的输入维度不同。
        具体来说，我们传递了一个包含 3 个时间步的输入，但神经网络期望的输入维度只有 2（即每个时间步有 2 个输入值）。
        因此，当输入脉冲的维度超过期望的维度时，应该触发 panic。
    */

    let mut snn = DynSnnBuilder::new(2)  
        .add_layer(  // 向 SNN 添加一层神经元，并设置神经元属性、输入权重和内部连接权重
            vec![  // 第一层神经元的属性，包含 2 个神经元，每个神经元使用 LifNeuron 模型
                LifNeuron::new(0.3, 0.05, 0.84, 1.0, 1.0),  // 第一个神经元的属性
                LifNeuron::new(0.3, 0.87, 0.12, 0.89, 1.0),  // 第二个神经元的属性
            ], 
            vec![  // 第一层神经元的外部输入权重
                vec![0.12, 0.5],  // 输入信号 1 和 2 对神经元的影响
                vec![0.53, 0.43], // 输入信号 1 和 2 对神经元的影响
            ], 
            vec![  // 神经元之间的内部连接权重（层内连接）
                vec![0.0, -0.3],  // 神经元 1 与神经元 2 之间的连接权重
                vec![-0.4, 0.0]    // 神经元 2 与神经元 1 之间的连接权重
            ]
        ).build();  // 完成 SNN 网络的构建

    // 传入的输入脉冲包含 3 个时间步，但每个时间步都只有 2 个输入信号。
    // 第一个时间步的输入是 [0, 0]，第二个时间步是 [0, 1]，第三个时间步是 [1, 1]。
    // 然而，这个 SNN 的设计只接受每个时间步 2 个输入信号，因此输入脉冲的维度（3x2）超过了网络的输入维度。
    // 由于输入的时间步数多于神经网络期望的维度，这应该导致 panic。
    let _output_spikes = snn.process(&vec![vec![0, 0], vec![0, 1], vec![1, 1]]);
}


#[test]  
#[should_panic] 
fn test_dyn_snn_input_spikes_lower_than_input_dimension() {  
    #[rustfmt::skip]  

    /*
        该测试用例需要触发 panic，因为输入脉冲的维度与神经网络的输入维度不同。
        具体来说，传递的输入脉冲只有一个时间步，并且每个时间步只有 2 个输入信号。
        然而，神经网络的期望输入维度是 2 个时间步，每个时间步有 2 个输入信号。
        因此，输入的时间步数比神经网络期望的少，这应该导致 panic。
    */

    let mut snn = DynSnnBuilder::new(2)  
        .add_layer(  // 向 SNN 添加一层神经元，并设置神经元属性、输入权重和内部连接权重
            vec![  // 第一层神经元的属性，包含 2 个神经元，每个神经元使用 LifNeuron 模型
                LifNeuron::new(0.3, 0.05, 0.84, 1.0, 1.0),  // 第一个神经元的属性
                LifNeuron::new(0.3, 0.87, 0.12, 0.89, 1.0),  // 第二个神经元的属性
            ], 
            vec![  // 第一层神经元的外部输入权重
                vec![0.12, 0.5],  // 输入信号 1 和 2 对神经元的影响
                vec![0.53, 0.43], // 输入信号 1 和 2 对神经元的影响
            ], 
            vec![  // 神经元之间的内部连接权重（层内连接）
                vec![0.0, -0.3],  // 神经元 1 与神经元 2 之间的连接权重
                vec![-0.4, 0.0]    // 神经元 2 与神经元 1 之间的连接权重
            ]
        ).build();  // 完成 SNN 网络的构建

    // 传入的输入脉冲包含 1 个时间步，并且每个时间步有 2 个输入信号
    // 但此时神经网络期望每个时间步有 2 个输入信号，且有 2 个时间步，因此输入的时间步数不足，导致维度不匹配。
    // 这个输入脉冲的维度是 1x2，而网络期望的维度是 2x2，因此应该触发 panic。
    let _output_spikes = snn.process(&vec![vec![1, 0]]);
}

