use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

//与 SNN 流畅构建器相关的测试

// 这个函数用于验证 LIF 神经元的各个参数是否符合期望值。
// 它比较了神经元的属性与给定的期望值，确保它们匹配。
// 如果所有参数匹配，则返回 true；否则返回 false。

fn verify_neuron(  // 定义函数，接受一个引用类型的 LifNeuron 和一些期望的参数。
    lif_neuron: &LifNeuron,  // LifNeuron 类型的引用，用来表示待验证的神经元实例。
    v_th: f64,               // 期望的阈值电压（膜电位阈值）。
    v_rest: f64,             // 期望的静息电压（膜电位的休息状态）。
    v_reset: f64,            // 期望的重置电压（神经元重置后的电压值）。
    tau: f64,                // 期望的时间常数 tau（决定了神经元膜电位的变化速度）。
    dt: f64                  // 期望的时间步长（时间增量，用于模拟）。
) -> bool { 
   
    if lif_neuron.get_v_th() != v_th {
        return false;  
    }

    if lif_neuron.get_v_rest() != v_rest {
        return false; 
    }

    if lif_neuron.get_v_reset() != v_reset {
        return false; 
    }

    if lif_neuron.get_tau() != tau {
        return false;  
    }

    if lif_neuron.get_dt() != dt {
        return false;  
    }

    // 检查神经元的膜电位（v_mem）是否与静息电压（v_rest）匹配。
    // 这是因为神经元初始化时，膜电位通常应该等于静息电压。
    if lif_neuron.get_v_mem() != v_rest {
        return false;
    }

    // 检查神经元的时间戳（ts）是否为 0。
    // 这个值通常用于记录神经元的更新时间，初始时应该为 0。
    if lif_neuron.get_ts() != 0u64 {
        return false;
    }

    true 
}


#[test]
fn test_add_one_layer() {  
    #[rustfmt::skip]  

    let snn = SnnBuilder::<LifNeuron>::new() 
        .add_layer::<0>()  // 向 SNN 中添加一个层，指定层的数量为 0
        .weights([])  // 设置神经网络的外部输入权重为空（没有输入权重）
        .neurons([])  // 设置神经网络的神经元为空（没有神经元）
        .intra_weights([])  // 设置神经网络的层内连接权重为空（没有内部连接权重）
        .build();  // 完成 SNN 网络的构建

    assert_eq!(snn.get_layers_number(), 1);  
}


#[test]
fn test_add_more_than_one_layer() {
    #[rustfmt::skip]

        let snn = SnnBuilder::<LifNeuron>::new()
        .add_layer::<0>().weights([]).neurons([]).intra_weights([])
        .add_layer().weights([]).neurons([]).intra_weights([])
        .add_layer().weights([]).neurons([]).intra_weights([])
        .add_layer().weights([]).neurons([]).intra_weights([])
        .build();

    assert_eq!(snn.get_layers_number(),4);
}

#[test]
fn test_add_weights_to_layers() { 
    #[rustfmt::skip]

    let snn_params = SnnBuilder::<LifNeuron>::new() 
        .add_layer()  // 向神经网络中添加第一层
        .weights([  // 为第一层指定外部输入权重
            [0.1, 0.2, 0.3],  // 第一层的权重矩阵（3 个输入，3 个神经元）
            [0.4, 0.5, 0.6],  // 第二层的权重矩阵（3 个输入，3 个神经元）
        ])
        .neurons([  // 为第一层指定神经元
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第一个神经元，使用 LifNeuron 构造函数
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 第二个神经元
        ])
        .intra_weights([  // 为第一层设置层内连接权重
            [0.0, -0.2],  // 第一层的层内连接权重矩阵（2 个神经元之间的连接）
            [-0.9, 0.0],   // 第二层的层内连接权重矩阵（2 个神经元之间的连接）
        ])
        .add_layer()  // 向神经网络中添加第二层
        .weights([  // 为第二层指定外部输入权重
            [0.2, 0.3],  // 第二层的权重矩阵（2 个输入，1 个神经元）
        ])
        .neurons([  // 为第二层指定神经元
            LifNeuron::new(0.45, 0.7, 0.1, 0.6, 1.0),  // 第三个神经元
        ])
        .intra_weights([  // 为第二层设置层内连接权重
            [0.0],  // 第一层的层内连接权重矩阵（1 个神经元）
        ])
        .get_params();  // 获取当前构建的 SNN 的所有参数

    // 获取每一层的外部输入权重矩阵
    let weights_layer1 = snn_params.extra_weights.get(0);  // 获取第一层的权重
    let weights_layer2 = snn_params.extra_weights.get(1);  // 获取第二层的权重
    let weights_layer3 = snn_params.extra_weights.get(2);  // 获取第三层的权重

    // 检查第一层和第二层的权重是否存在（即是否正确设置了权重）
    assert_eq!(weights_layer1.is_some(), true);  
    assert_eq!(weights_layer2.is_some(), true);  
    assert_eq!(weights_layer3.is_none(), true);  // 确保第三层的权重不存在（因为没有第三层）

    // 检查第一层和第二层的权重是否与期望值相等
    assert_eq!(weights_layer1.unwrap(), &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);  // 验证第一层的权重
    assert_eq!(weights_layer2.unwrap(), &[[0.2, 0.3]]);  // 验证第二层的权重
}


#[test]
fn test_layer_with_one_neuron() { 
    #[rustfmt::skip] 

    let snn_params = SnnBuilder::new() 
        .add_layer()  
        .weights([  // 为当前层指定外部输入权重矩阵
            [0.3, 0.5, 0.1, 0.6, 0.3]  // 输入权重矩阵（包含 5 个输入特征，1 个神经元）
        ])
        .neurons([  // 为当前层指定神经元
            LifNeuron::new(0.12, 0.8, 0.03, 0.64, 1.0)  // 添加一个神经元，使用 LifNeuron 构造函数
        ])
        .intra_weights([  // 为当前层设置层内连接权重矩阵
            [0.0]  // 由于只有一个神经元，所以层内权重矩阵只有一个值
        ])
        .get_params();  // 获取当前构建的 SNN 的所有参数（神经元、权重等）

    // 获取第一层的神经元集合
    let layer_neurons1 = snn_params.neurons.get(0);  // 获取第一层的神经元（应该只有一个神经元）
    let layer_neurons2 = snn_params.neurons.get(1);  // 尝试获取第二层的神经元（应为空）

    // 断言验证第一层是否包含神经元，并且该层神经元数量为 1
    assert_eq!(layer_neurons1.is_some(), true);  // 确保第一层有神经元
    assert_eq!(layer_neurons1.unwrap().len(), 1);  // 确保第一层只有 1 个神经元
    assert_eq!(layer_neurons2.is_none(), true);  // 确保没有第二层神经元

    // 获取并验证第一层的神经元参数
    let neuron = layer_neurons1.unwrap().get(0); 

    // 断言验证该神经元存在，并且该神经元的参数符合预期
    assert_eq!(neuron.is_some(), true);  // 确保神经元存在
    assert_eq!(verify_neuron(neuron.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);  // 验证神经元的各个参数
}


#[test]
fn test_neurons_with_same_parameters1() {  // 测试名称：测试具有相同参数的神经元
    #[rustfmt::skip] 

    let snn_params = SnnBuilder::new() 
        .add_layer() 
        .weights([  // 为该层设置外部输入权重矩阵
            [0.3, 0.5, 0.1, 0.6, 0.3],  // 第一行权重
            [0.2, 0.3, 0.1, 0.4, 0.2]   // 第二行权重
        ])
        .neurons([  // 为该层添加神经元，添加两个具有相同参数的 LIF 神经元
            LifNeuron::new(0.12, 0.8, 0.03, 0.64, 1.0),  // 第一个神经元
            LifNeuron::new(0.12, 0.8, 0.03, 0.64, 1.0)   // 第二个神经元（参数相同）
        ])
        .intra_weights([  // 设置该层的内连接权重矩阵
            [0.0, -0.3],  // 第一个神经元到第二个神经元的权重
            [-0.2, 0.0]   // 第二个神经元到第一个神经元的权重
        ])
        .get_params();  // 获取构建的神经网络参数（神经元、权重、内连接权重）

    // 获取该层的神经元集合
    let layer_neurons1 = snn_params.neurons.get(0);  // 获取第一层神经元集合（该层包含两个神经元）

    assert_eq!(layer_neurons1.is_some(), true);  // 确保第一层有神经元
    assert_eq!(layer_neurons1.unwrap().len(), 2);  // 确保该层有 2 个神经元

    let neuron1 = layer_neurons1.unwrap().get(0);  // 获取第一个神经元
    let neuron2 = layer_neurons1.unwrap().get(1);  // 获取第二个神经元

    assert_eq!(neuron1.is_some(), true);  // 确保第一个神经元存在
    assert_eq!(neuron2.is_some(), true);  // 确保第二个神经元存在

    assert_eq!(verify_neuron(neuron1.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);  // 验证第一个神经元参数
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);  // 验证第二个神经元参数
}


#[test]
fn test_neurons_with_same_parameters2() { 
    #[rustfmt::skip] 

    // 使用 SnnBuilder 构建一个神经网络层，并使用 `neurons_with_same_parameters` 方法来添加 2 个具有相同参数的神经元
    let snn_params = SnnBuilder::new() 
        .add_layer()  
        .weights([  // 为该层设置外部输入权重矩阵
            [0.3, 0.5, 0.1, 0.6, 0.3],  // 第一行权重
            [0.2, 0.3, 0.1, 0.4, 0.2]   // 第二行权重
        ])
        .neurons_with_same_parameters(  // 使用相同的参数创建 2 个神经元并添加到该层
            LifNeuron::new(0.12, 0.8, 0.03, 0.64, 1.0),  // 神经元参数
            2  // 创建 2 个具有相同参数的神经元
        )
        .intra_weights([  // 设置该层的内连接权重矩阵
            [0.0, -0.3],  // 第一个神经元到第二个神经元的权重
            [-0.2, 0.0]   // 第二个神经元到第一个神经元的权重
        ])
        .get_params();  // 获取构建的神经网络参数（神经元、权重、内连接权重）

    // 获取该层的神经元集合
    let layer_neurons1 = snn_params.neurons.get(0);  // 获取第一层神经元集合（该层包含两个神经元）

    assert_eq!(layer_neurons1.is_some(), true);  // 确保第一层有神经元
    assert_eq!(layer_neurons1.unwrap().len(), 2);  // 确保该层有 2 个神经元

    let neuron1 = layer_neurons1.unwrap().get(0);  // 获取第一个神经元
    let neuron2 = layer_neurons1.unwrap().get(1);  // 获取第二个神经元

    assert_eq!(neuron1.is_some(), true);  // 确保第一个神经元存在
    assert_eq!(neuron2.is_some(), true);  // 确保第二个神经元存在

    assert_eq!(verify_neuron(neuron1.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);  // 验证第一个神经元参数
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);  // 验证第二个神经元参数
}


#[test]
fn test_layer_with_more_than_one_neuron() { 
    #[rustfmt::skip] 

    let snn_params = SnnBuilder::new() 
        .add_layer() 
        .weights([  // 为该层的神经元设置外部输入权重矩阵
            [0.3, 0.5, 0.1, 0.6, 0.3],  // 第一个神经元的输入权重
            [0.2, 0.3, 0.1, 0.9, 0.76], // 第二个神经元的输入权重
            [0.1, 0.2, 0.3, 0.4, 0.5]   // 第三个神经元的输入权重
        ])
        .neurons([  // 向该层添加 3 个神经元，并为每个神经元设置不同的参数
            LifNeuron::new(0.127, 0.46, 0.78, 0.67, 1.0),  // 第一个神经元的参数
            LifNeuron::new(0.12, 0.22, 0.31, 0.47, 1.0),   // 第二个神经元的参数
            LifNeuron::new(0.25, 0.36, 0.5, 0.84, 1.0)     // 第三个神经元的参数
        ])
        .intra_weights([  // 设置神经元之间的内连接权重矩阵
            [0.0, -0.34, -0.12],  // 第一层神经元的内连接权重
            [-0.23, 0.0, -0.56],   // 第二层神经元的内连接权重
            [-0.05, -0.01, 0.0]    // 第三层神经元的内连接权重
        ])
        .get_params();  // 获取神经网络的参数（神经元、外部权重、内连接权重）

    // 获取该层的神经元集合（此层有 3 个神经元）
    let layer_neurons1 = snn_params.neurons.get(0);  // 获取第一层神经元集合（包含 3 个神经元）
    let layer_neurons2 = snn_params.neurons.get(1);

    // 断言：确认第一层确实包含 3 个神经元
    assert_eq!(layer_neurons1.is_some(), true);  // 确保第一层有神经元
    assert_eq!(layer_neurons1.unwrap().len(), 3);  // 确保第一层包含 3 个神经元
    assert_eq!(layer_neurons2.is_none(), true);  // 确保没有第二层（因为只添加了一个层）

    // 获取并验证第一个神经元的参数
    let neuron1 = layer_neurons1.unwrap().get(0);  // 获取第一层的第一个神经元
    assert_eq!(neuron1.is_some(), true);  // 确保第一个神经元存在
    assert_eq!(verify_neuron(neuron1.unwrap(), 0.127, 0.46, 0.78, 0.67, 1.0), true);  // 验证神经元的参数是否正确

    // 获取并验证第二个神经元的参数
    let neuron2 = layer_neurons1.unwrap().get(1);  // 获取第一层的第二个神经元
    assert_eq!(neuron2.is_some(), true);  // 确保第二个神经元存在
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.22, 0.31, 0.47, 1.0), true);  // 验证神经元的参数是否正确

    // 获取并验证第三个神经元的参数
    let neuron3 = layer_neurons1.unwrap().get(2);  // 获取第一层的第三个神经元
    assert_eq!(neuron3.is_some(), true);  // 确保第三个神经元存在
    assert_eq!(verify_neuron(neuron3.unwrap(), 0.25, 0.36, 0.5, 0.84, 1.0), true);  // 验证神经元的参数是否正确
}


#[test]
fn test_intra_layer_weights_with_one_neuron() { 
    #[rustfmt::skip] 

    let snn_params = SnnBuilder::new()  // 创建一个新的 SNN 构建器实例
        .add_layer()  // 向神经网络中添加一层神经元
        .weights([  // 为该层的神经元设置外部输入权重矩阵
            [0.3, 0.5, 0.1, 0.6, 0.3]  // 为该层的唯一神经元设置输入权重
        ])
        .neurons([  // 向该层添加一个神经元，并为该神经元设置参数
            LifNeuron::new(0.12, 0.1, 0.03, 0.98, 1.0)  // 创建一个 LifNeuron 神经元，并设置其阈值、休息电位、重置电位、时间常数和时间步长
        ])
        .intra_weights([  // 设置该层的内连接权重矩阵，当前层只有一个神经元，因此内连接权重只有一个值
            [0.0]  // 第一个神经元的内连接权重矩阵，只有一个值 0.0
        ])
        .get_params();  // 获取构建好的神经网络的所有参数，包括神经元、外部权重和内连接权重等

    // 获取该层的内连接权重集合（此层只有一个神经元，因此只有一个内连接权重矩阵）
    let layer_intra_weights1 = snn_params.intra_weights.get(0);  // 获取第一层的内连接权重集合（包含一个矩阵）
    let layer_intra_weights2 = snn_params.intra_weights.get(1);  // 获取第二层的内连接权重集合（应该为空）

    // 断言：确保该层包含 1 个内连接权重矩阵
    assert_eq!(layer_intra_weights1.is_some(), true);  // 确保第一层有内连接权重
    assert_eq!(layer_intra_weights1.unwrap().len(), 1);  // 确保该内连接权重矩阵的长度为 1（因为只有一个神经元）
    assert_eq!(layer_intra_weights2.is_none(), true);  // 确保没有第二层（因为只有一层）

    // 获取并验证第一个内连接权重矩阵的内容
    let intra_weights = layer_intra_weights1.unwrap().get(0);  // 获取第一层的第一个内连接权重矩阵
    assert_eq!(intra_weights.is_some(), true);  // 确保内连接权重矩阵存在
    assert_eq!(intra_weights.unwrap(), &[0.0]);  // 验证该内连接权重矩阵的内容是 [0.0]
}


#[test]
fn test_intra_layer_weights_with_more_than_one_neuron() {  
    #[rustfmt::skip] 

    let snn_params = SnnBuilder::new() 
        .add_layer()  // 向神经网络中添加一层神经元
        .weights([  // 为该层的神经元设置外部输入权重矩阵
            [0.3, 0.5, 0.1, 0.6, 0.3],  // 为第一个神经元设置输入权重
            [0.2, 0.3, 0.1, 0.9, 0.76],  // 为第二个神经元设置输入权重
            [0.1, 0.2, 0.3, 0.4, 0.5]    // 为第三个神经元设置输入权重
        ])
        .neurons([  // 向该层添加多个神经元，并为这些神经元设置各自的参数
            LifNeuron::new(0.127, 0.12, 0.78, 0.67, 1.0),  // 第一个神经元
            LifNeuron::new(0.12, 0.22, 0.31, 0.47, 1.0),   // 第二个神经元
            LifNeuron::new(0.25, 0.36, 0.71, 0.84, 1.0)    // 第三个神经元
        ])
        .intra_weights([  // 设置该层的内连接权重矩阵
            [0.0, -0.34, -0.12],  // 第一个神经元的内连接权重
            [-0.23, 0.0, -0.56],  // 第二个神经元的内连接权重
            [-0.05, -0.01, 0.0]   // 第三个神经元的内连接权重
        ])
        .get_params();  // 获取构建好的神经网络的所有参数，包括神经元、外部权重和内连接权重等

    // 获取该层的内连接权重集合
    let layer_intra_weights1 = snn_params.intra_weights.get(0);  // 获取第一层的内连接权重集合
    let layer_intra_weights2 = snn_params.intra_weights.get(1);  // 获取第二层的内连接权重集合（应该为空）

    // 断言：确保第一层包含 3 个内连接权重矩阵（因为有 3 个神经元）
    assert_eq!(layer_intra_weights1.is_some(), true);  // 确保第一层有内连接权重
    assert_eq!(layer_intra_weights1.unwrap().len(), 3);  // 确保该内连接权重集合包含 3 个权重矩阵
    assert_eq!(layer_intra_weights2.is_none(), true);  // 确保没有第二层（因为只有一层）

    // 获取并验证第一组内连接权重矩阵的内容
    let weights1 = layer_intra_weights1.unwrap().get(0);  // 获取第一组内连接权重矩阵
    assert_eq!(weights1.is_some(), true);  // 确保该权重矩阵存在
    assert_eq!(weights1.unwrap(), &[0.0, -0.34, -0.12]);  // 验证该内连接权重矩阵的内容是 [0.0, -0.34, -0.12]

    // 获取并验证第二组内连接权重矩阵的内容
    let weights2 = layer_intra_weights1.unwrap().get(1);  // 获取第二组内连接权重矩阵
    assert_eq!(weights2.is_some(), true);  // 确保该权重矩阵存在
    assert_eq!(weights2.unwrap(), &[-0.23, 0.0, -0.56]);  // 验证该内连接权重矩阵的内容是 [-0.23, 0.0, -0.56]

    // 获取并验证第三组内连接权重矩阵的内容
    let weights3 = layer_intra_weights1.unwrap().get(2);  // 获取第三组内连接权重矩阵
    assert_eq!(weights3.is_some(), true);  // 确保该权重矩阵存在
    assert_eq!(weights3.unwrap(), &[-0.05, -0.01, 0.0]);  // 验证该内连接权重矩阵的内容是 [-0.05, -0.01, 0.0]
}


#[test]
fn test_complete_snn_with_different_dt() {  // 验证不同时间步长 (dt) 下的 SNN 完整性
    #[rustfmt::skip] 
    
    let dt = 0.1;  // 设置神经元的时间步长 dt 为 0.1

    // 使用 SnnBuilder 构建一个完整的脉冲神经网络（SNN），并添加两层神经元
    let snn = SnnBuilder::new()  // 创建一个新的 SNN 构建器实例
        .add_layer()  
        .weights([  // 为第一层的神经元设置外部输入权重矩阵
            [0.9, 0.42, 0.1, 0.31, 0.3],  // 第一个神经元的输入权重
            [0.2, 0.56, 0.1, 0.9, 0.76],  // 第二个神经元的输入权重
            [0.2, 0.23, 0.3, 0.95, 0.5],  // 第三个神经元的输入权重
            [0.23, 0.1, 0.2, 0.4, 0.8]    // 第四个神经元的输入权重
        ])
        .neurons([  // 向第一层添加 4 个神经元，并为这些神经元设置各自的参数（包括时间步长 dt）
            LifNeuron::new(0.1, 0.1, 0.23, 0.45, dt),  // 第一个神经元
            LifNeuron::new(0.3, 0.12, 0.54, 0.23, dt),  // 第二个神经元
            LifNeuron::new(0.2, 0.23, 0.23, 0.65, dt),  // 第三个神经元
            LifNeuron::new(0.4, 0.34, 0.12, 0.45, dt)   // 第四个神经元
        ])
        .intra_weights([  // 设置第一层的内连接权重矩阵
            [0.0, -0.34, -0.12, -0.23],  // 第一个神经元的内连接权重
            [-0.23, 0.0, -0.56, -0.23],  // 第二个神经元的内连接权重
            [-0.05, -0.01, 0.0, -0.23],  // 第三个神经元的内连接权重
            [-0.23, -0.23, -0.23, 0.0]   // 第四个神经元的内连接权重
        ])
        .add_layer()  // 向神经网络中添加第二层神经元
        .weights([  // 为第二层的神经元设置外部输入权重矩阵
            [0.1, 0.3, 0.4, 0.2],  // 第一个神经元的输入权重
            [0.7, 0.3, 0.1, 0.3]   // 第二个神经元的输入权重
        ])
        .neurons([  // 向第二层添加 2 个神经元，并为这些神经元设置各自的参数（包括时间步长 dt）
            LifNeuron::new(0.17, 0.12, 0.78, 0.67, dt),  // 第一个神经元
            LifNeuron::new(0.25, 0.36, 0.71, 0.84, dt)   // 第二个神经元
        ])
        .intra_weights([  // 设置第二层的内连接权重矩阵
            [0.0, -0.62],  // 第一个神经元的内连接权重
            [-0.12, 0.0]   // 第二个神经元的内连接权重
        ])
        .build();  // 构建神经网络并返回该神经网络实例

    // 获取神经网络的所有层
    let snn_layers = snn.get_layers();  // 获取网络中的所有层

    // 断言：确保神经网络包含 2 层
    assert_eq!(snn_layers.len(), 2);  // 网络应该包含 2 层

    // 获取第一层和第二层
    let layer1 = snn_layers.get(0);  // 获取第一层
    let layer2 = snn_layers.get(1);  // 获取第二层
    let layer3 = snn_layers.get(2);  // 尝试获取第三层（应该为 None）

    // 断言：确保第一层和第二层存在，而第三层不存在
    assert_eq!(layer1.is_some(), true);  // 第一层应该存在
    assert_eq!(layer2.is_some(), true);  // 第二层应该存在
    assert_eq!(layer3.is_none(), true);  // 第三层应该为空（因为只添加了 2 层）

    // 获取第一层的神经元、权重和内连接权重
    let neurons_layer1 = layer1.unwrap().get_neurons();  // 获取第一层的神经元列表
    let weights_layer1 = layer1.unwrap().get_weights();  // 获取第一层的输入权重矩阵
    let intra_weights_layer1 = layer1.unwrap().get_intra_weights();  // 获取第一层的内连接权重矩阵

    // 断言：第一层的神经元、权重和内连接权重数量应为 4
    assert_eq!(neurons_layer1.len(), 4);  // 第一层包含 4 个神经元
    assert_eq!(weights_layer1.len(), 4);  // 第一层包含 4 行输入权重
    assert_eq!(intra_weights_layer1.len(), 4);  // 第一层包含 4 行内连接权重

    // 验证第一层的每个神经元的参数是否符合预期
    assert_eq!(verify_neuron(&neurons_layer1[0], 0.1, 0.1, 0.23, 0.45, dt), true);  // 验证第一个神经元
    assert_eq!(verify_neuron(&neurons_layer1[1], 0.3, 0.12, 0.54, 0.23, dt), true);  // 验证第二个神经元
    assert_eq!(verify_neuron(&neurons_layer1[2], 0.2, 0.23, 0.23, 0.65, dt), true);  // 验证第三个神经元
    assert_eq!(verify_neuron(&neurons_layer1[3], 0.4, 0.34, 0.12, 0.45, dt), true);  // 验证第四个神经元

    // 断言：第一层的输入权重和内连接权重符合预期
    assert_eq!(weights_layer1, &[  // 验证第一层的输入权重矩阵
        [0.9, 0.42, 0.1, 0.31, 0.3],
        [0.2, 0.56, 0.1, 0.9, 0.76],
        [0.2, 0.23, 0.3, 0.95, 0.5],
        [0.23, 0.1, 0.2, 0.4, 0.8]
    ]);
    assert_eq!(intra_weights_layer1, &[  // 验证第一层的内连接权重矩阵
        [0.0, -0.34, -0.12, -0.23],
        [-0.23, 0.0, -0.56, -0.23],
        [-0.05, -0.01, 0.0, -0.23],
        [-0.23, -0.23, -0.23, 0.0]
    ]);

    // 获取第二层的神经元、权重和内连接权重
    let neurons_layer2 = layer2.unwrap().get_neurons();  // 获取第二层的神经元列表
    let weights_layer2 = layer2.unwrap().get_weights();  // 获取第二层的输入权重矩阵
    let intra_weights_layer2 = layer2.unwrap().get_intra_weights();  // 获取第二层的内连接权重矩阵

    // 断言：第二层的神经元、权重和内连接权重数量应为 2
    assert_eq!(neurons_layer2.len(), 2);  // 第二层包含 2 个神经元
    assert_eq!(weights_layer2.len(), 2);  // 第二层包含 2 行输入权重
    assert_eq!(intra_weights_layer2.len(), 2);  // 第二层包含 2 行内连接权重

    // 验证第二层的每个神经元的参数是否符合预期
    assert_eq!(verify_neuron(&neurons_layer2[0], 0.17, 0.12, 0.78, 0.67, dt), true);  // 验证第一个神经元
    assert_eq!(verify_neuron(&neurons_layer2[1], 0.25, 0.36, 0.71, 0.84, dt), true);  // 验证第二个神经元

    // 断言：第二层的输入权重和内连接权重符合预期
    assert_eq!(weights_layer2, &[  // 验证第二层的输入权重矩阵
        [0.1, 0.3, 0.4, 0.2],
        [0.7, 0.3, 0.1, 0.3]
    ]);
    assert_eq!(intra_weights_layer2, &[  // 验证第二层的内连接权重矩阵
        [0.0, -0.62],
        [-0.12, 0.0]
    ]);

    // 断言：确认神经网络的层数为 2
    assert_eq!(snn.get_layers_number(), 2);  // 网络应该包含 2 层
}


#[test]
fn test_complete_snn() {  // 验证完整的 SNN（脉冲神经网络）构建过程
    #[rustfmt::skip]  

    let snn = SnnBuilder::new() 
        .add_layer() 
        .weights([  // 设置第一层的外部输入权重矩阵
            [0.9, 0.42, 0.1, 0.31, 0.3],  // 第一个神经元的输入权重
            [0.2, 0.56, 0.1, 0.9, 0.76],  // 第二个神经元的输入权重
            [0.2, 0.23, 0.3, 0.95, 0.5],  // 第三个神经元的输入权重
            [0.23, 0.1, 0.2, 0.4, 0.8]    // 第四个神经元的输入权重
        ])
        .neurons([  // 向第一层添加 4 个神经元，并为这些神经元设置各自的参数（包括时间步长 dt）
            LifNeuron::new(0.1, 0.1, 0.23, 0.45, 1.0),  // 第一个神经元
            LifNeuron::new(0.3, 0.12, 0.54, 0.23, 1.0),  // 第二个神经元
            LifNeuron::new(0.2, 0.23, 0.23, 0.65, 1.0),  // 第三个神经元
            LifNeuron::new(0.4, 0.34, 0.12, 0.45, 1.0)   // 第四个神经元
        ])
        .intra_weights([  // 设置第一层的内连接权重矩阵
            [0.0, -0.34, -0.12, -0.23],  // 第一个神经元的内连接权重
            [-0.23, 0.0, -0.56, -0.23],  // 第二个神经元的内连接权重
            [-0.05, -0.01, 0.0, -0.23],  // 第三个神经元的内连接权重
            [-0.23, -0.23, -0.23, 0.0]   // 第四个神经元的内连接权重
        ])
        .add_layer()  // 向神经网络中添加第二层神经元
        .weights([  // 设置第二层的外部输入权重矩阵
            [0.1, 0.3, 0.4, 0.2],  // 第一个神经元的输入权重
            [0.7, 0.3, 0.1, 0.3]   // 第二个神经元的输入权重
        ])
        .neurons([  // 向第二层添加 2 个神经元，并为这些神经元设置各自的参数（包括时间步长 dt）
            LifNeuron::new(0.17, 0.12, 0.78, 0.67, 1.0),  // 第一个神经元
            LifNeuron::new(0.25, 0.36, 0.71, 0.84, 1.0)   // 第二个神经元
        ])
        .intra_weights([  // 设置第二层的内连接权重矩阵
            [0.0, -0.62],  // 第一个神经元的内连接权重
            [-0.12, 0.0]   // 第二个神经元的内连接权重
        ])
        .build();  // 构建神经网络并返回该神经网络实例

    // 获取神经网络的所有层
    let snn_layers = snn.get_layers();  // 获取网络中的所有层

    // 断言：确保神经网络包含 2 层
    assert_eq!(snn_layers.len(), 2);  // 网络应该包含 2 层

    // 获取第一层和第二层
    let layer1 = snn_layers.get(0);  // 获取第一层
    let layer2 = snn_layers.get(1);  // 获取第二层
    let layer3 = snn_layers.get(2);  // 尝试获取第三层（应该为 None）

    // 断言：确保第一层和第二层存在，而第三层不存在
    assert_eq!(layer1.is_some(), true);  // 第一层应该存在
    assert_eq!(layer2.is_some(), true);  // 第二层应该存在
    assert_eq!(layer3.is_none(), true);  // 第三层应该为空（因为只添加了 2 层）

    // 获取第一层的神经元、权重和内连接权重
    let neurons_layer1 = layer1.unwrap().get_neurons();  // 获取第一层的神经元列表
    let weights_layer1 = layer1.unwrap().get_weights();  // 获取第一层的输入权重矩阵
    let intra_weights_layer1 = layer1.unwrap().get_intra_weights();  // 获取第一层的内连接权重矩阵

    assert_eq!(neurons_layer1.len(), 4);  // 第一层包含 4 个神经元
    assert_eq!(weights_layer1.len(), 4);  // 第一层包含 4 行输入权重
    assert_eq!(intra_weights_layer1.len(), 4);  // 第一层包含 4 行内连接权重

    assert_eq!(verify_neuron(&neurons_layer1[0], 0.1, 0.1, 0.23, 0.45, 1.0), true);  // 验证第一个神经元
    assert_eq!(verify_neuron(&neurons_layer1[1], 0.3, 0.12, 0.54, 0.23, 1.0), true);  // 验证第二个神经元
    assert_eq!(verify_neuron(&neurons_layer1[2], 0.2, 0.23, 0.23, 0.65, 1.0), true);  // 验证第三个神经元
    assert_eq!(verify_neuron(&neurons_layer1[3], 0.4, 0.34, 0.12, 0.45, 1.0), true);  // 验证第四个神经元

    assert_eq!(weights_layer1, &[  // 验证第一层的输入权重矩阵
        [0.9, 0.42, 0.1, 0.31, 0.3],
        [0.2, 0.56, 0.1, 0.9, 0.76],
        [0.2, 0.23, 0.3, 0.95, 0.5],
        [0.23, 0.1, 0.2, 0.4, 0.8]
    ]);
    assert_eq!(intra_weights_layer1, &[  // 验证第一层的内连接权重矩阵
        [0.0, -0.34, -0.12, -0.23],
        [-0.23, 0.0, -0.56, -0.23],
        [-0.05, -0.01, 0.0, -0.23],
        [-0.23, -0.23, -0.23, 0.0]
    ]);

    // 获取第二层的神经元、权重和内连接权重
    let neurons_layer2 = layer2.unwrap().get_neurons();  // 获取第二层的神经元列表
    let weights_layer2 = layer2.unwrap().get_weights();  // 获取第二层的输入权重矩阵
    let intra_weights_layer2 = layer2.unwrap().get_intra_weights();  // 获取第二层的内连接权重矩阵

    // 断言：第二层的神经元、权重和内连接权重数量应为 2
    assert_eq!(neurons_layer2.len(), 2);  // 第二层包含 2 个神经元
    assert_eq!(weights_layer2.len(), 2);  // 第二层包含 2 行输入权重
    assert_eq!(intra_weights_layer2.len(), 2);  // 第二层包含 2 行内连接权重

    // 验证第二层的每个神经元的参数是否符合预期
    assert_eq!(verify_neuron(&neurons_layer2[0], 0.17, 0.12, 0.78, 0.67, 1.0), true);  // 验证第一个神经元
    assert_eq!(verify_neuron(&neurons_layer2[1], 0.25, 0.36, 0.71, 0.84, 1.0), true);  // 验证第二个神经元

    assert_eq!(weights_layer2, &[  // 验证第二层的输入权重矩阵
        [0.1, 0.3, 0.4, 0.2],
        [0.7, 0.3, 0.1, 0.3]
    ]);
    assert_eq!(intra_weights_layer2, &[  // 验证第二层的内连接权重矩阵
        [0.0, -0.62],
        [-0.12, 0.0]
    ]);

    assert_eq!(snn.get_layers_number(), 2);  // 网络应该包含 2 层
}


#[test] 
#[should_panic] 
fn test_snn_with_negative_weights() {  // 验证 SNN 构建时包含负权重的情况，是否会触发 panic
    #[rustfmt::skip] 

    let _snn = SnnBuilder::new()  
        .add_layer() 
        .weights([  // 设置该层的外部输入权重矩阵
            [-0.2, 0.5]  // 权重包含负值（-0.2），可能违反网络设计要求，期望触发 panic
        ])
        .neurons([  // 向该层添加一个神经元，并为该神经元设置各自的参数（包括时间步长 dt）
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)  // 创建一个 LIF 神经元，设置其参数
        ])
        .intra_weights([  // 设置该层的内连接权重矩阵
            [0.0]  // 没有内连接的权重（仅有一个神经元）
        ])
        .build();  // 构建神经网络并返回该神经网络实例
}


#[test] 
#[should_panic] 
fn test_snn_with_positive_intra_weights() {  // 验证 SNN 构建时包含正的内连接权重的情况，是否会触发 panic
    #[rustfmt::skip] 

    // 使用 SnnBuilder 构建一个脉冲神经网络（SNN），并设置神经元和权重，包括正的内连接权重
    let _snn = SnnBuilder::new()  
        .add_layer()  
        .weights([  // 设置该层的外部输入权重矩阵
            [0.2, 0.5],  // 第一行输入权重：0.2, 0.5
            [0.3, 0.4]   // 第二行输入权重：0.3, 0.4
        ])
        .neurons([  // 向该层添加神经元，并为每个神经元设置参数
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),  // 创建第一个 LIF 神经元，设置其参数
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)   // 创建第二个 LIF 神经元，设置其参数
        ])
        .intra_weights([  // 设置该层的内连接权重矩阵
            [0.0, 0.5],  // 第一个神经元与第二个神经元之间的内连接权重为 0.5
            [-0.05, 0.0]  // 第二个神经元与第一个神经元之间的内连接权重为 -0.05
        ])
        .build();  // 构建神经网络并返回该神经网络实例
}
