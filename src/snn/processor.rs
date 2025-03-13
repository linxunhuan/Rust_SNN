use crate::neuron::Neuron;
use crate::snn::layer::Layer;
use crate::SpikeEvent;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

#[derive(Debug)]
pub struct Processor {}

impl Processor {
    /**
       Spikes 是一个包含 spike 事件的向量，这些事件将通过网络的层进行处理
       - 此方法为每一层创建一个新线程
       每个线程将通过共享通道处理从上一层接收到的输入 spike 事件，并使用另一个共享通道将计算得到的输出 spike 事件发送到下一层
    */
    pub fn process_events<
        'a,
        N: Neuron + Clone + Send + 'static,
        S: IntoIterator<Item = &'a mut Arc<Mutex<Layer<N>>>>,
    >(
        &self,
        snn: S,
        spikes: Vec<SpikeEvent>,
    ) -> Vec<SpikeEvent> {
        /* 创建线程池 */
        let mut threads = Vec::<JoinHandle<()>>::new();

        /* 创建通道以向网络（第一层）提供输入 */
        let (net_input_tx, mut layer_rc) = channel::<SpikeEvent>();

        /* 为每一层创建输入 TX 和输出 RC 并生成层的线程 */
        for layer_ref in snn {
            /* 创建通道以向下一层提供输入 */
            let (layer_tx, next_layer_rc) = channel::<SpikeEvent>();

            let layer_ref = layer_ref.clone();

            let thread = thread::spawn(move || {
                /* 获取层 */
                let mut layer = layer_ref.lock().unwrap();
                /* 执行层的任务 */
                layer.process(layer_rc, layer_tx);
            });

            threads.push(thread); /* 将新线程加入线程池 */
            layer_rc = next_layer_rc; /* 更新外部 rc，以将其传递到下一层 */
        }

        let net_output_rc = layer_rc;

        /* 将输入 SpikeEvents 发送到 *net_input_tx* */
        for spike_event in spikes {
            /* * 检查是否至少有一个 spike，否则跳过到下一个时刻 * */
            if spike_event.spikes.iter().all(|spike| *spike == 0u8) {
                continue; /* （仅处理 *有效* 的 spike 事件） */
            }

            let instant = spike_event.ts;

            net_input_tx.send(spike_event).expect(&format!(
                "在发送输入 spike 事件 t={} 时出现意外错误",
                instant
            ));
        }

        drop(net_input_tx); /* 删除输入 tx，以使所有线程终止 */

        /* 从 *net_output* rc 获取输出 SpikeEvents */
        let mut output_events = Vec::<SpikeEvent>::new();

        while let Ok(spike_event) = net_output_rc.recv() {
            output_events.push(spike_event);
        }

        /* 等待线程终止 */
        for thread in threads {
            thread.join().unwrap();
        }

        output_events
    }
}
