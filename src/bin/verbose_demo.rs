mod demo_internals;
use pds_snn::builders::{DynSnnBuilder, SnnBuilder};
use pds_snn::models::neuron::lif::LifNeuron;
use crate::demo_internals::demo_internals::{print_instants, print_layer, print_spikes};


fn main() {
    // 打印信息，表明这是一个详细的脉冲神经网络（Spiking Neural Network）Rust库演示
    println!("* Verbose demo for the Spiking Neural Network Rust library *");

    /* DynSnnBuilder 和 DynSNN 的详细演示 */
    verbose_demo_dynamic_snn();

    // 打印一个空行，以便格式更整齐
    println!();

    /* SnnBuilder 和 SNN 的详细演示 */
    verbose_demo_static_snn();
}

fn verbose_demo_dynamic_snn() {
    println!("\n• *Dynamic* building demo");

    // 创建一个DynSnnBuilder对象，输入层有5个神经元
    let mut builder = DynSnnBuilder::<LifNeuron>::new(5);
    println!("Created Builder for *dynamic* SNN with 5 input neurons");

    /* 添加第一层（隐藏层） */
    builder = builder.add_layer(vec![
        /* 4个LIF神经元 */
        LifNeuron::new(0.1, 0.10, 0.23, 0.45, 1.0),
        LifNeuron::new(0.3, 0.12, 0.54, 0.23, 1.0),
        LifNeuron::new(0.2, 0.23, 0.23, 0.65, 1.0),
        LifNeuron::new(0.4, 0.34, 0.12, 0.45, 1.0)
    ], vec![
        vec![0.9 , 0.42, 0.1, 0.31, 0.3 ],      /* 1号神经元从输入层获取的额外权重 */
        vec![0.2 , 0.56, 0.1, 0.9 , 0.76],      /* 2号神经元从输入层获取的额外权重 */
        vec![0.2 , 0.23, 0.3, 0.95, 0.5 ],      /* 3号神经元从输入层获取的额外权重 */
        vec![0.23, 0.1 , 0.2, 0.4 , 0.8 ]       /* 4号神经元从输入层获取的额外权重 */
    ], vec![
        vec![ 0.0 , -0.34, -0.12, -0.23],       /* 1号神经元的内部权重 */
        vec![-0.23,  0.0 , -0.56, -0.23],       /* 2号神经元的内部权重 */
        vec![-0.05, -0.01,  0.0 , -0.23],       /* 3号神经元的内部权重 */
        vec![-0.23, -0.23, -0.23,  0.0 ]        /* 4号神经元的内部权重 */
    ]);

    println!("\nAdded 1st hidden layer with 4 LifNeurons:");
    print_layer(&builder.get_params().neurons[0]);

    /* 添加第二层（隐藏层） */
    builder = builder.add_layer(vec![
        /* 2个LIF神经元 */
        LifNeuron::new(0.17, 0.12, 0.78, 0.67, 1.0),
        LifNeuron::new(0.25, 0.36, 0.71, 0.84, 1.0)
    ], vec![
        vec![0.1, 0.3, 0.4, 0.2],   /* 1号神经元从第一层获取的额外权重 */
        vec![0.7, 0.3, 0.1, 0.3],   /* 2号神经元从第一层获取的额外权重 */
    ], vec![
        vec![ 0.0 , -0.62],         /* 1号神经元的内部权重 */
        vec![-0.12,  0.0 ],         /* 2号神经元的内部权重 */
    ]);

    println!("\nAdded 2nd hidden layer with 2 LifNeurons:");
    print_layer(&builder.get_params().neurons[1]);

    println!("\nBuilding the network...");
    // 构建脉冲神经网络
    let mut snn = builder.build();
    println!("Done!");


    println!("\n• Dynamic processing");

    /* 创建输入脉冲 */
    let input_spikes = vec![
        vec![1, 0, 1, 1, 0, 0, 1, 0, 0, 1],     /* 1号神经元的输入脉冲序列 */
        vec![0, 0, 1, 1, 1, 0, 1, 1, 0, 1],     /* 2号神经元的输入脉冲序列 */
        vec![0, 1, 0, 1, 0, 0, 1, 0, 0, 0],     /* 3号神经元的输入脉冲序列 */
        vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 0],     /* 4号神经元的输入脉冲序列 */
        vec![1, 1, 1, 0, 0, 0, 1, 0, 0, 1]      /* 5号神经元的输入脉冲序列 */
    ];

    println!("在10个时间点内的输入脉冲：");
    print_instants(input_spikes[0].len());
    print_spikes(&input_spikes, "input");

    /* 处理输入脉冲 */
    println!("\n将输入脉冲提供给DynSNN...");
    let output_spikes = snn.process(&input_spikes);
    println!("Done!");

    println!("\n脉冲神经网络生成的输出脉冲：");
    print_instants(output_spikes[0].len());
    print_spikes(&output_spikes, "output");
}

fn verbose_demo_static_snn() {
    println!("\n• *Static* building demo");

    // 创建一个静态脉冲神经网络（SNN）构建器
    let builder = SnnBuilder::<LifNeuron>::new();
    println!("Created Builder for *static* SNN");

    /* 添加第一层（隐藏层） */
    let builder = builder.add_layer()
        .weights([
            [0.9 , 0.42, 0.1, 0.31, 0.3 ],      /* 1号神经元从输入层获取的额外权重 */
            [0.2 , 0.56, 0.1, 0.9 , 0.76],      /* 2号神经元从输入层获取的额外权重 */
            [0.2 , 0.23, 0.3, 0.95, 0.5 ],      /* 3号神经元从输入层获取的额外权重 */
            [0.23, 0.1 , 0.2, 0.4 , 0.8 ]       /* 4号神经元从输入层获取的额外权重 */
        ]).neurons([
            /* 4个LIF神经元 */
            LifNeuron::new(0.1, 0.10, 0.23, 0.45, 1.0),
            LifNeuron::new(0.3, 0.12, 0.54, 0.23, 1.0),
            LifNeuron::new(0.2, 0.23, 0.23, 0.65, 1.0),
            LifNeuron::new(0.4, 0.34, 0.12, 0.45, 1.0)
        ]).intra_weights([
            [ 0.0 , -0.34, -0.12, -0.23],       /* 1号神经元的内部权重 */
            [-0.23,  0.0 , -0.56, -0.23],       /* 2号神经元的内部权重 */
            [-0.05, -0.01,  0.0 , -0.23],       /* 3号神经元的内部权重 */
            [-0.23, -0.23, -0.23,  0.0 ]        /* 4号神经元的内部权重 */
        ]);

    println!("\nAdded 1st hidden layer with 4 LifNeurons:");
    print_layer(&builder.get_params().neurons[0]);

    /* 添加第二层（隐藏层） */
    let builder = builder.add_layer()
        .weights([
            [0.1, 0.3, 0.4, 0.2],   /* 1号神经元从第一层获取的额外权重 */
            [0.7, 0.3, 0.1, 0.3]    /* 2号神经元从第一层获取的额外权重 */
        ]).neurons([
            /* 2个LIF神经元 */
            LifNeuron::new(0.17, 0.12, 0.78, 0.67, 1.0),
            LifNeuron::new(0.25, 0.36, 0.71, 0.84, 1.0)
        ]).intra_weights([
            [ 0.0 , -0.62],         /* 1号神经元的内部权重 */
            [-0.12,  0.0 ],         /* 2号神经元的内部权重 */
        ]);

    println!("\nAdded 2nd hidden layer with 2 LifNeurons:");
    print_layer(&builder.get_params().neurons[1]);

    println!("\nBuilding the network...");
    // 构建静态脉冲神经网络（SNN）
    let mut snn = builder.build();
    println!("Done!");

    println!("\n• Static processing");

    /* 创建输入脉冲 */
    let input_spikes = [
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],     /* 1号神经元的输入脉冲序列 */
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],     /* 2号神经元的输入脉冲序列 */
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],     /* 3号神经元的输入脉冲序列 */
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 0],     /* 4号神经元的输入脉冲序列 */
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]      /* 5号神经元的输入脉冲序列 */
    ];

    println!("在10个时间点内的输入脉冲：");
    print_instants(input_spikes[0].len());
    print_spikes(&input_spikes, "input");

    /* 处理输入脉冲 */
    println!("\n将输入脉冲提供给SNN...");
    let output_spikes = snn.process(&input_spikes);
    println!("Done!");

    println!("\n脉冲神经网络生成的输出脉冲：");
    print_instants(output_spikes[0].len());
    print_spikes(&output_spikes, "output");
}
