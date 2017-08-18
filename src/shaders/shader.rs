use std::path::Path;
use std::fs::File;
use std::sync::Arc;
use std::result::Result;
use vulkano::device::Device;
use vulkano::pipeline::shader::ShaderModule;
use utils::error::Error;
use vulkano::OomError;

pub struct Shader {
    device: Arc<Device>,
    vert_shader_module: Arc<ShaderModule>,
    frag_shader_module: Arc<ShaderModule>,
}

impl Shader {
    pub fn new(device: Arc<Device>, vert_shader_filepath: &Path, frag_shader_file_path: &Path) -> Shader {
        let vert_shader_code = read_file(vert_shader_filepath);
        let frag_shader_code = read_file(frag_shader_file_path);

        let vert_shader_module = create_shader_module(device.clone(), vert_shader_code).expect("Failed to load vertex shader");
        let frag_shader_module = create_shader_module(device.clone(), frag_shader_code).expect("Failed to load fragment shader");

        use vulkano;

        Shader {
            device: device,
            vert_shader_module: vert_shader_module,
            frag_shader_module: frag_shader_module,
        }
    }
}


fn create_shader_module(device: Arc<Device>, code: Vec<u8>) -> Result<Arc<ShaderModule>, Error> {
    let result = unsafe { ShaderModule::new(device, code.as_ref()) };

    let c = result.map_err(|err| {
        match err {
            OomError::OutOfDeviceMemory => Error::new("OutOfDeviceMemory".to_string()),
            OomError::OutOfHostMemory => Error::new("OutOfHostMemory".to_string()),
        }
    });

    return c;
}

fn read_file(file_path: &Path) -> Vec<u8> {
    use std::io::Read;

    let mut file = File::open(file_path).unwrap();

    let mut contents: Vec<u8> = Vec::new();
    // Returns amount of bytes read and append the result to the buffer
    let result = file.read_to_end(&mut contents).expect(format!("Error reading file {}", file_path.display()).as_str());

    contents
}
