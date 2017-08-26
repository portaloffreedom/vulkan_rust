use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::device::Queue;
use vulkano::format;
//use vulkano::image::ImageAccess;
use vulkano::image::ImmutableImage;
use vulkano::sync::NowFuture;

use shaders::shader::Shader;

pub struct Material {
    texture: Arc<ImmutableImage<format::R8G8B8A8Srgb>>, //ImageAccess
    shader: Arc<Shader>,
}

impl Material {
    pub fn new<P: AsRef<Path>>(queue: &Arc<Queue>, shader: Arc<Shader>, path: P)
                               -> Result<(Self, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>), String>
    {
        use vulkano::image::Dimensions;
        use image;

        let file = File::open(&path)
            .map_err(|e| format!("Error loading image file: {} - Error: {}", path.as_ref().display(), e))?;
        let file = BufReader::new(file);

//        let image = image::load_from_memory_with_format(include_bytes!("../../resources/GroundForest003_1k/GroundForest003_COL_VAR1_1K.jpg"),
//                                                        image::ImageFormat::JPEG)
//            .map_err(|e| format!("Error loading image: {}", e))?
//            .to_rgba();

        let image = image::load(file, image::ImageFormat::JPEG)
            .map_err(|e| format!("Error loading image: {}", e))?
            .to_rgba();

        let width = image.width();
        let height = image.height();
        let image_data = image.into_raw().clone();

        let (image, future) = ImmutableImage::from_iter(
            image_data.iter().cloned(),
            Dimensions::Dim2d { width: width, height: height },
            format::R8G8B8A8Srgb,
            queue.clone()
        ).map_err(|e| format!("Error creating the image: {}", e))?;

        Ok((Material {
            texture: image,
            shader: shader,
        }, future))
    }
}