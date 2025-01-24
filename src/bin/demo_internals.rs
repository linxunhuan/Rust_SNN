
pub mod demo_internals {
    use std::fmt::Debug;
    
    // 打印时间点
    pub fn print_instants(n: usize) {
        print!(" t\t ");
        (0..n).into_iter().for_each(|t| print!("{}  ", t));
        println!();
    }

    // 打印脉冲
    pub fn print_spikes<'a, S: IntoIterator<Item=K>, K: IntoIterator<Item=&'a u8> + Debug>(spikes: S, role: &str) {
        spikes.into_iter()
            .zip(vec!["1st", "2nd", "3rd", "4th", "5th"].into_iter())  // 将脉冲与位置标签配对
            .for_each(|(train_of_spikes, pos)|
                println!("\t{:?} \t\t {} *{}* 神经元的脉冲序列", train_of_spikes, pos, role));
    }

    // 打印层中的神经元
    #[allow(dead_code)]
    pub fn print_layer<L: IntoIterator<Item=N>, N: Debug>(neurons: L) {
        neurons.into_iter().for_each(|neuron| println!("- {:?}", neuron));  // 打印每个神经元
        println!("已添加对应的层间权重和层内权重");
    }
}

#[allow(dead_code)]
fn main() {}

