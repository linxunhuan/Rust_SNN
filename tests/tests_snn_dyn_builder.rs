use pds_snn::builders::DynSnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

//与 SNN 动态生成器相关的测试

// 该函数用于验证 LIF 神经元（Leaky Integrate-and-Fire Neuron）的参数是否正确。
// 它通过将传入的参数与神经元的当前值进行比较，确保神经元的参数设置正确。
// 如果所有参数匹配，则返回 `true`；否则返回 `false`。
fn verify_neuron(lif_neuron: &LifNeuron, v_th: f64, v_rest: f64, v_reset: f64, tau: f64, dt: f64) -> bool {
    
    if lif_neuron.get_v_th() != v_th {
        return false; 
    }

    if lif_neuron.get_v_rest() != v_rest {
        return false; 
    }

    if lif_neuron.get_v_reset() != v_reset {
        return false; 
    }

    if  lif_neuron.get_tau() != tau {
        return false;
    }

    if lif_neuron.get_dt() != dt {
        return false; 
    }

    if lif_neuron.get_v_mem() != v_rest {
        return false;  
    }

    if lif_neuron.get_ts() != 0u64 {
        return false; 
    }

    true
}


#[test]
fn test_add_one_layer() {
    // 使用 DynSnnBuilder 创建一个新的 SNN（动态神经网络）实例。
    // `LifNeuron` 是神经元类型，`0` 是网络的初始状态值（通常为时间步或网络的初始标识符）。
    // 使用 `add_layer` 添加一个新层，该层没有神经元、权重和内部连接权重（vec![] 是空向量）。
    let snn = DynSnnBuilder::<LifNeuron>::new(0)
        .add_layer(vec![], vec![], vec![])  // 向网络添加一个空的层
        .build();  // 构建网络

    // 验证：检查网络的层数是否为 1
    // 因为我们只添加了一个层，所以 get_layers_number() 应该返回 1。
    assert_eq!(snn.get_layers_number(), 1);
}


#[test]
fn test_add_more_than_one_layer() {
    let snn = DynSnnBuilder::<LifNeuron>::new(0)
        .add_layer(vec![], vec![], vec![])  // 添加第 1 个空层
        .add_layer(vec![], vec![], vec![])  // 添加第 2 个空层
        .add_layer(vec![], vec![], vec![])  // 添加第 3 个空层
        .add_layer(vec![], vec![], vec![])  // 添加第 4 个空层
        .build();  // 构建网络

    // 验证：检查网络的层数是否为 4
    // 因为我们添加了 4 个空层，所以 get_layers_number() 应该返回 4
    assert_eq!(snn.get_layers_number(), 4);
}

#[test]
// 该测试确保向每一层添加的权重是正确的，并能通过访问网络的层权重来验证
fn test_add_weights_to_layers() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new(3)
        .add_layer(
            vec![
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ], 
            // 向第一个层添加连接权重
            vec![
                vec![0.1, 0.2, 0.3], 
                vec![0.4, 0.5, 0.6]
            ], 
            // 向第一个层添加内部连接权重
            vec![
                vec![0.0, -0.2], 
                vec![-0.9, 0.0]
            ]
        )
        // 向第二个层添加 1 个神经元
        .add_layer(
            vec![
                LifNeuron::new(0.45, 0.7, 0.1, 0.6, 1.0)
            ], 
            // 向第二个层添加连接权重
            vec![
                vec![0.2, 0.3]
            ], 
            // 向第二个层添加内部连接权重
            vec![
                vec![0.0]
            ]
        )
        // 构建并获取神经网络的参数
        .get_params();

    // 获取三个层的连接权重
    let weights_layer1 = snn_params.extra_weights.get(0);
    let weights_layer2 = snn_params.extra_weights.get(1);
    let weights_layer3 = snn_params.extra_weights.get(2);

    assert_eq!(weights_layer1.is_some(), true);
    assert_eq!(weights_layer2.is_some(), true);
    assert_eq!(weights_layer3.is_none(), true);

    assert_eq!(weights_layer1.unwrap(), &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
    assert_eq!(weights_layer2.unwrap(), &[[0.2, 0.3]]);
}

// 该测试确保在神经网络中添加一个仅含有一个神经元的层时，能够正确地访问该神经元，并验证其参数
#[test]
fn test_layer_with_one_neuron() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new(5)
        .add_layer(
            vec![LifNeuron::new(0.12, 0.8, 0.03, 0.64, 1.0)], // 第一个层：包含一个神经元
            vec![vec![0.3, 0.5, 0.1, 0.6, 0.3]], // 权重矩阵
            vec![vec![0.0]] // 内部连接权重
        )
        .get_params(); // 获取网络参数

    let layer_neurons1 = snn_params.neurons.get(0); // 获取第一层神经元
    let layer_neurons2 = snn_params.neurons.get(1); // 获取第二层神经元

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 1);
    assert_eq!(layer_neurons2.is_none(), true);

    // 获取第一层的第一个神经元
    let neuron = layer_neurons1.unwrap().get(0);

    assert_eq!(neuron.is_some(), true);
    assert_eq!(verify_neuron(neuron.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);
}


// 本测试确保多个具有相同参数的神经元能够正确地添加到同一层，并且它们的参数也应当相同
#[test]
fn test_neurons_with_same_parameters1() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new(5)
        .add_layer_with_same_neurons(
            LifNeuron::new(0.12, 0.8, 0.03, 0.64, 1.0),  // 创建一个具有相同参数的神经元
            2,  // 添加 2 个相同的神经元
            vec![  // 层的权重矩阵
                vec![0.3, 0.5, 0.1, 0.6, 0.3],
                vec![0.2, 0.3, 0.1, 0.4, 0.2]
            ],
            vec![  // 层的内部权重矩阵
                vec![0.0, -0.3],
                vec![-0.2, 0.0]
            ]
        )
        .get_params();  // 获取网络的参数

    // 获取第一层的神经元列表
    let layer_neurons1 = snn_params.neurons.get(0);

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 2);

    let neuron1 = layer_neurons1.unwrap().get(0);  // 获取第一个神经元
    let neuron2 = layer_neurons1.unwrap().get(1);  // 获取第二个神经元

    assert_eq!(neuron1.is_some(), true);
    assert_eq!(neuron2.is_some(), true);

    assert_eq!(verify_neuron(neuron1.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);
}


// 本测试确保当多个神经元有相同的参数时，神经网络能够正确地处理并验证这些神经元的状态
#[test]
fn test_neurons_with_same_parameters2() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new(5)
        .add_layer(  
            vec![  
                LifNeuron::new(0.12, 0.8, 0.03, 0.64, 1.0),  // 第一个神经元
                LifNeuron::new(0.12, 0.8, 0.03, 0.64, 1.0),  // 第二个神经元
            ],
            vec![  // 权重矩阵
                vec![0.3, 0.5, 0.1, 0.6, 0.3],  // 第一个神经元的权重
                vec![0.2, 0.3, 0.1, 0.4, 0.2],  // 第二个神经元的权重
            ],
            vec![  // 内部连接权重矩阵
                vec![0.0, -0.3],  // 第一个神经元的内部权重
                vec![-0.2, 0.0],  // 第二个神经元的内部权重
            ]
        )
        .get_params();  // 获取网络的参数

    // 获取第一层的神经元列表
    let layer_neurons1 = snn_params.neurons.get(0);

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 2);

    let neuron1 = layer_neurons1.unwrap().get(0);  // 获取第一个神经元
    let neuron2 = layer_neurons1.unwrap().get(1);  // 获取第二个神经元

    assert_eq!(neuron1.is_some(), true);
    assert_eq!(neuron2.is_some(), true);

    assert_eq!(verify_neuron(neuron1.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.8, 0.03, 0.64, 1.0), true);
}


// 本测试确保当我们在同一层中添加多个神经元时，能够正确地访问并验证每个神经元的参数
#[test]
fn test_layer_with_more_than_one_neuron() {
    #[rustfmt::skip]

    // 使用 DynSnnBuilder 创建一个新的神经网络实例，添加一层具有 3 个神经元的层。
    // 每个神经元的参数：v_th, v_rest, v_reset, tau, dt
    // 为每个神经元定义了连接权重和内部连接权重
    let snn_params = DynSnnBuilder::<LifNeuron>::new(5)
        .add_layer(  
            vec![  
                LifNeuron::new(0.127, 0.46, 0.78, 0.67, 1.0),  // 第一个神经元
                LifNeuron::new(0.12, 0.22, 0.31, 0.47, 1.0),   // 第二个神经元
                LifNeuron::new(0.25, 0.36, 0.5, 0.84, 1.0),    // 第三个神经元
            ],
            vec![  // 权重矩阵
                vec![0.3, 0.5, 0.1, 0.6, 0.3],  // 第一个神经元的权重
                vec![0.2, 0.3, 0.1, 0.9, 0.76], // 第二个神经元的权重
                vec![0.1, 0.2, 0.3, 0.4, 0.5],  // 第三个神经元的权重
            ],
            vec![  // 内部连接权重矩阵
                vec![0.0, -0.34, -0.12],  // 第一个神经元的内部连接权重
                vec![-0.23, 0.0, -0.56],  // 第二个神经元的内部连接权重
                vec![-0.05, -0.01, 0.0],  // 第三个神经元的内部连接权重
            ]
        )
        .get_params();  // 获取网络的参数

    let layer_neurons1 = snn_params.neurons.get(0); // 第一层的神经元
    let layer_neurons2 = snn_params.neurons.get(1); // 第二层的神经元

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 3);
    assert_eq!(layer_neurons2.is_none(), true);

    // 获取并验证第一个神经元
    let neuron1 = layer_neurons1.unwrap().get(0); // 获取第一个神经元
    assert_eq!(neuron1.is_some(), true);  // 确保第一个神经元存在
    assert_eq!(verify_neuron(neuron1.unwrap(), 0.127, 0.46, 0.78, 0.67, 1.0), true); // 验证第一个神经元的参数

    // 获取并验证第二个神经元
    let neuron2 = layer_neurons1.unwrap().get(1); // 获取第二个神经元
    assert_eq!(neuron2.is_some(), true);  // 确保第二个神经元存在
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.22, 0.31, 0.47, 1.0), true); // 验证第二个神经元的参数

    // 获取并验证第三个神经元
    let neuron3 = layer_neurons1.unwrap().get(2); // 获取第三个神经元
    assert_eq!(neuron3.is_some(), true);  // 确保第三个神经元存在
    assert_eq!(verify_neuron(neuron3.unwrap(), 0.25, 0.36, 0.5, 0.84, 1.0), true); // 验证第三个神经元的参数
}

// 本测试确保在只有一个神经元的层中，内部连接权重（`intra_weights`）能够被正确初始化和访问
#[test]
fn test_intra_layer_weights_with_one_neuron() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new(5)
        .add_layer(vec![  
            LifNeuron::new(0.12, 0.1, 0.03, 0.98, 1.0)  // 添加一个神经元
        ], vec![  // 外部连接权重（`weights`）
            vec![0.3, 0.5, 0.1, 0.6, 0.3]
        ], vec![  // 内部连接权重（`intra_weights`）
            vec![0.0]  // 只有一个神经元的层，其内部连接权重设置为 [0.0]
        ])
        .get_params();  // 获取神经网络的参数

    let layer_intra_weights1 = snn_params.intra_weights.get(0);  // 获取第一层的内部连接权重
    let layer_intra_weights2 = snn_params.intra_weights.get(1);  // 获取第二层的内部连接权重

    assert_eq!(layer_intra_weights1.is_some(), true);
    assert_eq!(layer_intra_weights1.unwrap().len(), 1);
    assert_eq!(layer_intra_weights2.is_none(), true);

    let intra_weights = layer_intra_weights1.unwrap().get(0);  // 获取第一层的第一个内部连接权重

    assert_eq!(intra_weights.is_some(), true);
    assert_eq!(intra_weights.unwrap(), &[0.0]);
}


// 本测试确保在一个包含多个神经元的层中，神经元之间的内部连接权重（`intra_weights`）能够正确初始化和访问
#[test]
fn test_intra_layer_weights_with_more_than_one_neuron() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new(5)
        .add_layer(vec![  // 添加一层
            LifNeuron::new(0.127, 0.12, 0.78, 0.67, 1.0),  // 神经元1
            LifNeuron::new(0.12, 0.22, 0.31, 0.47, 1.0),  // 神经元2
            LifNeuron::new(0.25, 0.36, 0.71, 0.84, 1.0)   // 神经元3
        ], vec![  // 外部连接权重（`weights`）
            vec![0.3, 0.5, 0.1, 0.6, 0.3],  // 神经元1的外部连接权重
            vec![0.2, 0.3, 0.1, 0.9, 0.76],  // 神经元2的外部连接权重
            vec![0.1, 0.2, 0.3, 0.4, 0.5]   // 神经元3的外部连接权重
        ], vec![  // 内部连接权重（`intra_weights`）
            vec![0.0, -0.34, -0.12],  // 第一神经元的内部连接权重
            vec![-0.23, 0.0, -0.56],  // 第二神经元的内部连接权重
            vec![-0.05, -0.01, 0.0]   // 第三神经元的内部连接权重
        ])
        .get_params();  // 获取神经网络的参数

    let layer_intra_weights1 = snn_params.intra_weights.get(0);  // 第一层的内部连接权重
    let layer_intra_weights2 = snn_params.intra_weights.get(1);  // 第二层的内部连接权重（应为 None）

    assert_eq!(layer_intra_weights1.is_some(), true);
    assert_eq!(layer_intra_weights1.unwrap().len(), 3);
    assert_eq!(layer_intra_weights2.is_none(), true);

    // 获取并验证第一个神经元的内部连接权重
    let weights1 = layer_intra_weights1.unwrap().get(0);  // 获取第一神经元的内部连接权重
    assert_eq!(weights1.is_some(), true);  // 确保第一神经元的内部连接权重存在
    assert_eq!(weights1.unwrap(), &[0.0, -0.34, -0.12]);  // 验证内部连接权重是否正确

    // 获取并验证第二个神经元的内部连接权重
    let weights2 = layer_intra_weights1.unwrap().get(1);  // 获取第二神经元的内部连接权重
    assert_eq!(weights2.is_some(), true);  // 确保第二神经元的内部连接权重存在
    assert_eq!(weights2.unwrap(), &[-0.23, 0.0, -0.56]);  // 验证内部连接权重是否正确

    // 获取并验证第三个神经元的内部连接权重
    let weights3 = layer_intra_weights1.unwrap().get(2);  // 获取第三神经元的内部连接权重
    assert_eq!(weights3.is_some(), true);  // 确保第三神经元的内部连接权重存在
    assert_eq!(weights3.unwrap(), &[-0.05, -0.01, 0.0]);  // 验证内部连接权重是否正确
}


// 本测试检查构建的SNN中的每一层，确保神经元的数量、外部连接权重、以及内部连接权重的初始化是正确的
#[test]
fn test_complete_snn() {
    #[rustfmt::skip]

    // 使用 DynSnnBuilder 构建一个 SNN，其中包含两个层：
    // 第一个层包含 4 个神经元，第二个层包含 2 个神经元。
    let snn = DynSnnBuilder::<LifNeuron>::new(5)
        .add_layer(vec![  // 第一层神经元，包含 4 个神经元
            LifNeuron::new(0.1, 0.1, 0.23, 0.45, 1.0),  // 神经元 1
            LifNeuron::new(0.3, 0.12, 0.54, 0.23, 1.0), // 神经元 2
            LifNeuron::new(0.2, 0.23, 0.23, 0.65, 1.0), // 神经元 3
            LifNeuron::new(0.4, 0.34, 0.12, 0.45, 1.0)  // 神经元 4
        ], vec![  // 第一层的外部连接权重（权重矩阵）
            vec![0.9, 0.42, 0.1, 0.31, 0.3],  // 神经元 1 的连接权重
            vec![0.2, 0.56, 0.1, 0.9, 0.76],  // 神经元 2 的连接权重
            vec![0.2, 0.23, 0.3, 0.95, 0.5],  // 神经元 3 的连接权重
            vec![0.23, 0.1, 0.2, 0.4, 0.8]    // 神经元 4 的连接权重
        ], vec![  // 第一层的内部连接权重（权重矩阵）
            vec![0.0, -0.34, -0.12, -0.23],   // 神经元 1 的内部连接权重
            vec![-0.23, 0.0, -0.56, -0.23],   // 神经元 2 的内部连接权重
            vec![-0.05, -0.01, 0.0, -0.23],   // 神经元 3 的内部连接权重
            vec![-0.23, -0.23, -0.23, 0.0]    // 神经元 4 的内部连接权重
        ])
        .add_layer(vec![  // 第二层神经元，包含 2 个神经元
            LifNeuron::new(0.17, 0.12, 0.78, 0.67, 1.0),  // 神经元 5
            LifNeuron::new(0.25, 0.36, 0.71, 0.84, 1.0)   // 神经元 6
        ], vec![  // 第二层的外部连接权重
            vec![0.1, 0.3, 0.4, 0.2],  // 神经元 5 的连接权重
            vec![0.7, 0.3, 0.1, 0.3]   // 神经元 6 的连接权重
        ], vec![  // 第二层的内部连接权重
            vec![0.0, -0.62],  // 神经元 5 的内部连接权重
            vec![-0.12, 0.0]   // 神经元 6 的内部连接权重
        ])
        .build();  // 完成神经网络的构建

    // SNN应该包含 2 层
    let snn_layers = snn.get_layers();
    assert_eq!(snn_layers.len(), 2);

    // 获取并验证第一层和第二层的神经元、权重以及内部连接权重
    let layer1 = snn_layers.get(0);  // 第一层
    let layer2 = snn_layers.get(1);  // 第二层
    let layer3 = snn_layers.get(2);  // 不应该有第三层

    assert_eq!(layer1.is_some(), true);
    assert_eq!(layer2.is_some(), true);
    assert_eq!(layer3.is_none(), true);

    // 获取并验证第一层的神经元、权重和内部权重
    let neurons_layer1 = layer1.unwrap().get_neurons();
    let weights_layer1 = layer1.unwrap().get_weights();
    let intra_weights_layer1 = layer1.unwrap().get_intra_weights();

    // 断言：第一层有 4 个神经元，4 个外部连接权重，4 个内部连接权重
    assert_eq!(neurons_layer1.len(), 4);
    assert_eq!(weights_layer1.len(), 4);
    assert_eq!(intra_weights_layer1.len(), 4);

    // 验证第一层每个神经元的参数
    assert_eq!(verify_neuron(&neurons_layer1[0], 0.1, 0.1, 0.23, 0.45, 1.0), true);
    assert_eq!(verify_neuron(&neurons_layer1[1], 0.3, 0.12, 0.54, 0.23, 1.0), true);
    assert_eq!(verify_neuron(&neurons_layer1[2], 0.2, 0.23, 0.23, 0.65, 1.0), true);
    assert_eq!(verify_neuron(&neurons_layer1[3], 0.4, 0.34, 0.12, 0.45, 1.0), true);

    // 验证第一层的外部连接权重是否正确
    assert_eq!(weights_layer1, &[ 
        [0.9, 0.42, 0.1, 0.31, 0.3],
        [0.2, 0.56, 0.1, 0.9, 0.76],
        [0.2, 0.23, 0.3, 0.95, 0.5],
        [0.23, 0.1, 0.2, 0.4, 0.8]
    ]);

    // 验证第一层的内部连接权重是否正确
    assert_eq!(intra_weights_layer1, &[ 
        [0.0, -0.34, -0.12, -0.23],
        [-0.23, 0.0, -0.56, -0.23],
        [-0.05, -0.01, 0.0, -0.23],
        [-0.23, -0.23, -0.23, 0.0]
    ]);

    // 获取并验证第二层的神经元、权重和内部连接权重
    let neurons_layer2 = layer2.unwrap().get_neurons();
    let weights_layer2 = layer2.unwrap().get_weights();
    let intra_weights_layer2 = layer2.unwrap().get_intra_weights();

    // 断言：第二层有 2 个神经元，2 个外部连接权重，2 个内部连接权重
    assert_eq!(neurons_layer2.len(), 2);
    assert_eq!(weights_layer2.len(), 2);
    assert_eq!(intra_weights_layer2.len(), 2);

    // 验证第二层每个神经元的参数
    assert_eq!(verify_neuron(&neurons_layer2[0], 0.17, 0.12, 0.78, 0.67, 1.0), true);
    assert_eq!(verify_neuron(&neurons_layer2[1], 0.25, 0.36, 0.71, 0.84, 1.0), true);

    // 验证第二层的外部连接权重是否正确
    assert_eq!(weights_layer2, &[ 
        [0.1, 0.3, 0.4, 0.2],
        [0.7, 0.3, 0.1, 0.3]
    ]);

    // 验证第二层的内部连接权重是否正确
    assert_eq!(intra_weights_layer2, &[ 
        [0.0, -0.62],
        [-0.12, 0.0]
    ]);

    // 验证 SNN 的层数是否正确
    assert_eq!(snn.get_layers_number(), 2);
}


#[test]
// 验证在设置特定的 dt（时间步长）时，SNN 构建和神经元的参数（特别是时间步长）是否正确地初始化和应用
fn test_complete_snn_with_different_dt() {
    // 设置时间步长 dt，影响神经元更新的速率
    let dt = 0.1;

    // 构建一个动态脉冲神经网络（SNN），包含两层神经元
    let snn = DynSnnBuilder::<LifNeuron>::new(5)
        // 第一层：4个神经元，初始化每个神经元的膜电位、阈值、重置电位、泄漏常数、时间步长等参数
        .add_layer(vec![
            LifNeuron::new(0.1, 0.1, 0.23, 0.45, dt),
            LifNeuron::new(0.3, 0.12, 0.54, 0.23, dt),
            LifNeuron::new(0.2, 0.23, 0.23, 0.65, dt),
            LifNeuron::new(0.4, 0.34, 0.12, 0.45, dt)], 
            // 第一层的外部连接权重（每个神经元与下层神经元的连接）
            vec![
                vec![0.9, 0.42, 0.1, 0.31, 0.3],
                vec![0.2, 0.56, 0.1, 0.9, 0.76],
                vec![0.2, 0.23, 0.3, 0.95, 0.5],
                vec![0.23, 0.1, 0.2, 0.4, 0.8]
            ], 
            // 第一层的内部连接权重（每个神经元与同层神经元的连接）
            vec![
                vec![0.0, -0.34, -0.12, -0.23],
                vec![-0.23, 0.0, -0.56, -0.23],
                vec![-0.05, -0.01, 0.0, -0.23],
                vec![-0.23, -0.23, -0.23, 0.0]
            ]
        )
        // 第二层：2个神经元，同样初始化每个神经元的参数
        .add_layer(vec![
            LifNeuron::new(0.17, 0.12, 0.78, 0.67, dt),
            LifNeuron::new(0.25, 0.36, 0.71, 0.84, dt)], 
            // 第二层的外部连接权重
            vec![
                vec![0.1, 0.3, 0.4, 0.2],
                vec![0.7, 0.3, 0.1, 0.3]
            ],
            // 第二层的内部连接权重
            vec![
                vec![0.0, -0.62],
                vec![-0.12, 0.0]
            ]
        )
        // 完成网络的构建
        .build();

    // 获取SNN的所有层
    let snn_layers = snn.get_layers();

    // 验证网络应该有2层
    assert_eq!(snn_layers.len(), 2);

    // 获取并验证第一层、第二层、以及第三层（应为空） 
    let layer1 = snn_layers.get(0);
    let layer2 = snn_layers.get(1);
    let layer3 = snn_layers.get(2);

    assert_eq!(layer1.is_some(), true); // 第一层应该存在
    assert_eq!(layer2.is_some(), true); // 第二层应该存在
    assert_eq!(layer3.is_none(), true); // 第三层应该是None

    // 获取第一层的神经元、外部权重和内部权重
    let neurons_layer1 = layer1.unwrap().get_neurons();
    let weights_layer1 = layer1.unwrap().get_weights();
    let intra_weights_layer1 = layer1.unwrap().get_intra_weights();

    // 验证第一层有4个神经元
    assert_eq!(neurons_layer1.len(), 4);
    assert_eq!(weights_layer1.len(), 4);
    assert_eq!(intra_weights_layer1.len(), 4);

    // 使用预期的值验证每个神经元的参数
    assert_eq!(verify_neuron(&neurons_layer1[0], 0.1, 0.1, 0.23, 0.45, dt), true);
    assert_eq!(verify_neuron(&neurons_layer1[1], 0.3, 0.12, 0.54, 0.23, dt), true);
    assert_eq!(verify_neuron(&neurons_layer1[2], 0.2, 0.23, 0.23, 0.65, dt), true);
    assert_eq!(verify_neuron(&neurons_layer1[3], 0.4, 0.34, 0.12, 0.45, dt), true);

    // 验证第一层的外部连接权重与预期值一致
    assert_eq!(weights_layer1, &[
        [0.9, 0.42, 0.1, 0.31, 0.3],
        [0.2, 0.56, 0.1, 0.9, 0.76],
        [0.2, 0.23, 0.3, 0.95, 0.5],
        [0.23, 0.1, 0.2, 0.4, 0.8]
    ]);

    // 验证第一层的内部连接权重与预期值一致
    assert_eq!(intra_weights_layer1, &[
        [0.0, -0.34, -0.12, -0.23],
        [-0.23, 0.0, -0.56, -0.23],
        [-0.05, -0.01, 0.0, -0.23],
        [-0.23, -0.23, -0.23, 0.0]
    ]);

    // 获取第二层的神经元、外部权重和内部权重
    let neurons_layer2 = layer2.unwrap().get_neurons();
    let weights_layer2 = layer2.unwrap().get_weights();
    let intra_weights_layer2 = layer2.unwrap().get_intra_weights();

    // 验证第二层有2个神经元
    assert_eq!(neurons_layer2.len(), 2);
    assert_eq!(weights_layer2.len(), 2);
    assert_eq!(intra_weights_layer2.len(), 2);

    // 使用预期的值验证每个神经元的参数
    assert_eq!(verify_neuron(&neurons_layer2[0], 0.17, 0.12, 0.78, 0.67, dt), true);
    assert_eq!(verify_neuron(&neurons_layer2[1], 0.25, 0.36, 0.71, 0.84, dt), true);

    // 验证第二层的外部连接权重与预期值一致
    assert_eq!(weights_layer2, &[
        [0.1, 0.3, 0.4, 0.2],
        [0.7, 0.3, 0.1, 0.3]
    ]);

    // 验证第二层的内部连接权重与预期值一致
    assert_eq!(intra_weights_layer2, &[
        [0.0, -0.62],
        [-0.12, 0.0]
    ]);

    assert_eq!(snn.get_layers_number(), 2);
}


#[test]
#[should_panic] 
fn test_snn_with_negative_weights() {
    #[rustfmt::skip] 

    let _snn = DynSnnBuilder::<LifNeuron>::new(2) 
        // 添加第一层神经元及其外部权重和内部权重
        .add_layer(
            vec![ 
                // 第一层有 1 个神经元，参数如下：
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0) // 创建 LIF 神经元，包含电流阈值 (0.3)，突触时间常数 (0.05)，重置电位 (0.1)，漏电常数 (1.0)，时间步长 (1.0)
            ],
            vec![ 
                // 第一层神经元的外部权重，连接到下一层
                vec![-0.2, 0.5] // 为第一个神经元设置外部突触连接权重，这里第一个权重值为负数 (-0.2)，这是不合法的权重
            ],
            vec![ 
                // 第一层的内部权重，连接神经元之间
                vec![0.0] // 在本例中，内部权重没有连接（值为 0.0）
            ]
        )
        .build(); // 完成 SNN 的构建
}

#[test]
#[should_panic] 
fn test_dyn_snn_wrong_extra_weights1() {
    #[rustfmt::skip] 

    /*
        在这个测试中，额外权重 (extra weights) 是错误的，因为内部向量的列数
        必须等于上一层神经元的数量。即外部权重矩阵的列数应匹配前一层的神经元数量。
    */

    let _snn = DynSnnBuilder::<LifNeuron>::new(2) 
        // 添加第一层：包含 2 个神经元，设置每个神经元的 LIF 参数
        .add_layer(
            vec![ 
                // 第一层的 2 个神经元
                LifNeuron::new(0.1, 0.05, 0.1, 1.0, 1.0), // 第一个神经元的 LIF 参数
                LifNeuron::new(0.3, 0.23, 0.1, 0.89, 1.0) // 第二个神经元的 LIF 参数
            ],
            vec![ 
                // 第一层的外部权重，连接到下一层的神经元
                vec![0.2, 0.5], // 第一个神经元的外部权重
                vec![0.3, 0.4]  // 第二个神经元的外部权重
            ],
            vec![ 
                // 第一层的内部权重，连接神经元之间
                vec![0.0, -0.5], // 第一个神经元的内部连接权重
                vec![-0.05, 0.0] // 第二个神经元的内部连接权重
            ]
        )
        // 添加第二层：包含 1 个神经元，设置该神经元的 LIF 参数
        .add_layer(
            vec![ 
                LifNeuron::new(0.1, 0.05, 0.1, 1.0, 1.0) // 第二层的唯一神经元的 LIF 参数
            ],
            vec![ 
                // 第二层的外部权重，连接到上一层的神经元
                // 这里存在问题：上一层有 2 个神经元，因此权重的列数应该为 2。
                // 但是这里的权重有 3 个元素，因此会导致不匹配的情况，触发 panic。
                vec![0.3, 0.2, 0.4] // 错误的外部权重矩阵列数
            ],
            vec![ 
                // 第二层的内部权重
                vec![0.0] // 这里的权重也是有效的，但它不会影响外部权重的错误
            ]
        )
        .build(); // 当调用 build 时，由于外部权重的列数不匹配，会引发 panic
}


#[test]
#[should_panic]
fn test_dyn_snn_wrong_intra_weights1() {
    #[rustfmt::skip] 

    /*
        在这个测试中，内部权重（intra weights）是错误的，因为内部权重矩阵的列数
        必须等于当前层神经元的数量。即每个神经元的内部连接权重数应该等于当前层的神经元个数。
    */

    let _snn = DynSnnBuilder::<LifNeuron>::new(2) 
        // 添加第一层：包含 2 个神经元，设置每个神经元的 LIF 参数
        .add_layer(
            vec![ 
                // 第一层的 2 个神经元
                LifNeuron::new(0.1, 0.05, 0.1, 1.0, 1.0), // 第一个神经元的 LIF 参数
                LifNeuron::new(0.3, 0.23, 0.1, 0.89, 1.0) // 第二个神经元的 LIF 参数
            ],
            vec![ 
                // 第一层的外部权重，连接到下一层的神经元
                vec![0.2, 0.5], // 第一个神经元的外部权重
                vec![0.3, 0.4]  // 第二个神经元的外部权重
            ],
            vec![ 
                // 第一层的内部权重，连接神经元之间
                // 这里存在问题：第一层有 2 个神经元，因此内部权重的列数必须为 2。
                // 然而第一个神经元的内部连接权重有 3 个元素（vec![0.0, -0.1, -0.4]），
                // 这会导致列数不匹配，触发 panic。
                vec![0.0, -0.1, -0.4], // 第一个神经元的内部连接权重（错误）
                vec![-0.05, 0.0] // 第二个神经元的内部连接权重（正确，列数为 2）
            ]
        )
        .build(); 
}


#[test]
#[should_panic]
fn test_dyn_snn_wrong_extra_weights2() {
    #[rustfmt::skip] 
    /*
        在这个测试中，外部权重（extra weights）是错误的，因为外部权重矩阵的行数
        必须等于前一层神经元的数量。也就是说，每一层的外部连接权重的行数应当与上一层的神经元数量相等。
    */

    let _snn = DynSnnBuilder::<LifNeuron>::new(2) 
        // 添加第一层：包含 2 个神经元，设置每个神经元的 LIF 参数
        .add_layer(
            vec![ 
                // 第一层的 2 个神经元
                LifNeuron::new(0.1, 0.05, 0.1, 1.0, 1.0), // 第一个神经元的 LIF 参数
                LifNeuron::new(0.3, 0.23, 0.1, 0.89, 1.0) // 第二个神经元的 LIF 参数
            ],
            vec![ 
                // 第一层的外部权重，连接到下一层的神经元
                vec![0.2, 0.5], // 第一个神经元的外部权重
                vec![0.3, 0.4]  // 第二个神经元的外部权重
            ],
            vec![ 
                // 第一层的内部权重，连接神经元之间
                vec![0.0, -1.5], // 第一个神经元的内部连接权重
                vec![-0.05, 0.0] // 第二个神经元的内部连接权重
            ]
        )
        // 添加第二层：包含 1 个神经元，设置该神经元的 LIF 参数
        .add_layer(
            vec![ 
                LifNeuron::new(0.1, 0.05, 0.1, 1.0, 1.0) // 第二层的单个神经元
            ],
            vec![ 
                // 第二层的外部权重，连接到第三层的神经元
                // 这里存在问题：第一层有 2 个神经元，因此外部权重的行数必须为 2。
                // 然而，我们只为第二层的外部权重提供了 1 行（`vec![0.3, 0.2]` 和 `vec![0.1, 0.5]`），
                // 这将导致行数不匹配，触发 panic。
                vec![0.3, 0.2], // 第一个神经元的外部权重（错误，行数不匹配）
                vec![0.1, 0.5]  // 第二个神经元的外部权重（错误，行数不匹配）
            ],
            vec![ 
                // 第二层的内部权重，连接神经元之间
                vec![0.0] // 第二层只有一个神经元，因此内部权重矩阵只有一列
            ]
        )
        .build(); 
}


#[test]
#[should_panic] 
fn test_dyn_snn_wrong_intra_weights2() {
    #[rustfmt::skip] 
    /*
        在这个测试中，内部权重（intra weights）是错误的，因为内部权重矩阵的行数
        必须等于当前层神经元的数量。也就是说，每一层的神经元的数量应与该层的
        内部连接权重矩阵的行数相等。 
    */

    let _snn = DynSnnBuilder::<LifNeuron>::new(2) // 创建一个包含 2 层的动态 SNN 网络
        // 添加第一层：包含 2 个神经元，设置每个神经元的 LIF 参数
        .add_layer(
            vec![ 
                // 第一层的 2 个神经元
                LifNeuron::new(0.1, 0.05, 0.1, 1.0, 1.0), // 第一个神经元的 LIF 参数
                LifNeuron::new(0.3, 0.23, 0.1, 0.89, 1.0) // 第二个神经元的 LIF 参数
            ],
            vec![ 
                // 第一层的外部权重，连接到下一层的神经元
                vec![0.2, 0.5], // 第一个神经元的外部权重
                vec![0.3, 0.4]  // 第二个神经元的外部权重
            ],
            vec![ 
                // 第一层的内部权重，连接神经元之间
                // 这里有问题：第一层有 2 个神经元，因此内部权重矩阵的行数应该是 2。
                // 然而，我们为第一层提供了 3 行的内部权重（`vec![0.0, -0.1]`, `vec![-0.05, 0.0]`, `vec![0.0, -0.3]`），
                // 这将导致行数不匹配，触发 panic。
                vec![0.0, -0.1], // 第一个神经元的内部连接权重（错误，行数不匹配）
                vec![-0.05, 0.0], // 第二个神经元的内部连接权重（错误，行数不匹配）
                vec![0.0, -0.3]  // 第三个神经元的内部连接权重（错误，额外的行）
            ]
        )
        // 添加第二层：包含 1 个神经元，设置该神经元的 LIF 参数
        .add_layer(
            vec![ 
                LifNeuron::new(0.1, 0.05, 0.1, 1.0, 1.0) // 第二层的单个神经元
            ],
            vec![ 
                // 第二层的外部权重，连接到第三层的神经元
                vec![0.3, 0.2], // 第一个神经元的外部权重
                vec![0.1, 0.5]  // 第二个神经元的外部权重
            ],
            vec![ 
                // 第二层的内部权重，连接神经元之间
                vec![0.0] // 第二层只有一个神经元，因此内部权重矩阵只有一列
            ]
        )
        .build(); 
}


#[test]
#[should_panic] 
fn test_dyn_snn_with_zero_layers() {
    #[rustfmt::skip] 

    /*
        在这个测试中，我们尝试创建一个没有层的脉冲神经网络（SNN）。
        由于 SNN 至少需要一层来构建，因此如果尝试创建一个零层网络，
        我们期望构建过程会触发 panic。
    */

    // 使用 DynSnnBuilder 创建一个没有层的 SNN 网络，并调用 build() 方法构建它
    let _snn = DynSnnBuilder::<LifNeuron>::new(0) 
        .build(); 
}

