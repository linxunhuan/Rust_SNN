use byteorder::{BigEndian, ReadBytesExt};
use ndarray::{Array2, ArrayD};
use std::fs::File;
use std::io::{Cursor, Read};

// 数据类型解码器
fn decode_data_type(int_code: u32) -> String {
    match int_code {
        8 => "u8".to_string(),   // 无符号字节
        9 => "i8".to_string(),   // 有符号字节
        11 => "i16".to_string(), // 大端 16 位整数
        12 => "i32".to_string(), // 大端 32 位整数
        13 => "f32".to_string(), // 大端 32 位浮点数
        14 => "f64".to_string(), // 大端 64 位浮点数
        _ => panic!("未知的数据类型代码: {}", int_code),
    }
}

// 读取文件内容
fn read_file(filename: &str) -> Vec<u8> {
    let mut file = File::open(filename).expect("无法打开文件");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("无法读取文件");
    buffer
}

// 读取和解码魔术数
fn magic_number(buffer: &[u8]) -> (String, u8) {
    let mut cursor = Cursor::new(buffer);
    let mn = cursor.read_u32::<BigEndian>().expect("无法读取魔术数");
    let data_type_code = (mn >> 8) & 0xFF; // 提取数据类型字节
    let data_dim = (mn & 0xFF) as u8; // 提取维度数字节
    let dtype = decode_data_type(data_type_code);
    (dtype, data_dim)
}

// 读取维度
fn read_dimensions(buffer: &[u8], data_dim: u8) -> Vec<u32> {
    let mut cursor = Cursor::new(&buffer[4..]);
    let mut dimensions = Vec::new();
    for _ in 0..data_dim {
        let dim = cursor.read_u32::<BigEndian>().expect("无法读取维度");
        dimensions.push(dim);
    }
    dimensions
}

// 加载数据
fn load_data(buffer: &[u8], dtype: &str, offset: usize, dimensions: &[u32]) -> ArrayD<f64> {
    let total_elements = dimensions.iter().product::<u32>() as usize;
    let data_slice = &buffer[offset..];

    let data: Vec<f64> = match dtype {
        "u8" => data_slice.iter().map(|&x| x as f64).collect(),
        "i8" => data_slice.iter().map(|&x| x as i8 as f64).collect(),
        "i16" => {
            let mut cursor = Cursor::new(data_slice);
            (0..total_elements)
                .map(|_| cursor.read_i16::<BigEndian>().unwrap() as f64)
                .collect()
        }
        "i32" => {
            let mut cursor = Cursor::new(data_slice);
            (0..total_elements)
                .map(|_| cursor.read_i32::<BigEndian>().unwrap() as f64)
                .collect()
        }
        "f32" => {
            let mut cursor = Cursor::new(data_slice);
            (0..total_elements)
                .map(|_| cursor.read_f32::<BigEndian>().unwrap() as f64)
                .collect()
        }
        "f64" => {
            let mut cursor = Cursor::new(data_slice);
            (0..total_elements)
                .map(|_| cursor.read_f64::<BigEndian>().unwrap())
                .collect()
        }
        _ => panic!("不支持的数据类型: {}", dtype),
    };

    let shape: Vec<usize> = dimensions.iter().map(|&x| x as usize).collect();
    ArrayD::from_shape_vec(shape, data).expect("无法创建数组")
}

// 重塑数据
fn reshape_data(data: ArrayD<f64>, dimensions: &[u32]) -> ArrayD<f64> {
    if dimensions.len() > 1 {
        let array_dim = dimensions[1..].iter().product::<u32>() as usize;
        let shape = (dimensions[0] as usize, array_dim);
        data.into_shape(shape)
            .expect("重塑数组失败")
            .to_owned()
            .into_dyn()
    } else {
        data
    }
}

// 将 IDX 缓冲区转换为数组
fn idx_buffer_to_array(buffer: &[u8]) -> ArrayD<f64> {
    let (dtype, data_dim) = magic_number(buffer);
    let offset = 4 * data_dim as usize + 4; // 魔术数和维度后的数据偏移量
    let dimensions = read_dimensions(buffer, data_dim);
    let data = load_data(buffer, &dtype, offset, &dimensions);
    reshape_data(data, &dimensions)
}

// 加载数据集
pub fn load_dataset(images: &str, labels: &str) -> (Array2<f64>, ArrayD<f64>) {
    let img_buffer = read_file(images);
    let img_array = idx_buffer_to_array(&img_buffer);

    let labels_buffer = read_file(labels);
    let labels_array = idx_buffer_to_array(&labels_buffer);

    (
        img_array.into_dimensionality().expect("图像数组维度错误"),
        labels_array,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use ndarray::Axis;

    // 将 Array2<f64> 转换为 GrayImage
    fn array_to_image(array: &Array2<f64>) -> GrayImage {
        let (height, width) = array.dim();
        let mut img = GrayImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let pixel = array[[y, x]] as u8; // 将 f64 转换为 u8
                img.put_pixel(x as u32, y as u32, Luma([pixel]));
            }
        }
        img
    }

    // 展示 MNIST 图像
    fn show_mnist_image(image: &Array2<f64>) {
        let img = array_to_image(image);
        img.save("mnist_image.png").expect("无法保存图像");
        println!("图像已保存为 mnist_image.png");
    }

    #[test]
    fn test_load_mnist_image() {
        let images = "src/simulation/mnist/t10k-images-idx3-ubyte";
        let labels = "src/simulation/mnist/t10k-labels-idx1-ubyte";
        let (img_array, _labels_array) = load_dataset(images, labels);

        // 获取第一张图像，形状为 [784] 的一维数组
        let image_flat = img_array.index_axis(Axis(0), 0).to_owned();

        // 将一维数组重塑为 28x28 的二维数组
        let image = image_flat
            .into_shape((28, 28))
            .expect("无法将图像重塑为 28x28");

        // 调用函数，传递二维数组的引用
        show_mnist_image(&image);
    }
}
