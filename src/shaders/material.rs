use std::borrow::Borrow;
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::device::Queue;
use vulkano::device::Device;
use vulkano::format;
//use vulkano::image::ImageAccess;
use vulkano::image::ImmutableImage;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::sampler::Sampler;
use vulkano::sync::NowFuture;

use shaders::shader::Shader;

pub struct Material {
    texture: Arc<ImmutableImage<format::R8G8B8A8Srgb>>, //ImageAccess
    shader: Arc<Shader>,
}

impl Material {
    pub fn new<P: AsRef<Path>>(queue: &Arc<Queue>, shader: Arc<Shader>, path: P)
                               -> Result<(Arc<Self>, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>), String>
    {
        use vulkano::image::Dimensions;
        use image;

        let file = File::open(&path)
            .map_err(|e| format!("Error loading image file: {} - Error: {}", path.as_ref().display(), e))?;
        let file = BufReader::new(file);

        let extension = path.as_ref().extension()
            .ok_or(format!("No extension found on image file \"{}\"", path.as_ref().display()))?
            .to_string_lossy();

        let image_format = match extension.borrow() {
            "png" => image::ImageFormat::PNG,
            "jpg" => image::ImageFormat::JPEG,
            "tif" => image::ImageFormat::TIFF,
            _ => return Err(format!("\"{}\" is an unsupported image file \"{}\"!", extension, path.as_ref().display())),
        };

        let image = image::load(file, image_format)
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

        Ok(
            (Arc::new(Material {
                texture: image,
                shader: shader,
            }),
             future)
        )
    }

    fn sampler(&self, device: Arc<Device>) -> Result<Arc<Sampler>, String> {
        use vulkano::sampler::{Filter, MipmapMode, SamplerAddressMode};

        Sampler::new(device, Filter::Linear,
                     Filter::Linear, MipmapMode::Nearest,
                     SamplerAddressMode::ClampToEdge,
                     SamplerAddressMode::ClampToEdge,
                     SamplerAddressMode::ClampToEdge,
                     0.0, 1.0, 0.0, 0.0)
            .map_err(|e| format!("Error creating texture sampler: {}", e))
    }

    pub fn set(&self, device: Arc<Device>, pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>)
               -> Result<Arc<DescriptorSet + Send + Sync>, String>
    {
        let sampler = self.sampler(device)?;

        let set = PersistentDescriptorSet::start(pipeline, 0)
            .add_sampled_image(self.texture.clone(), sampler)
            .map_err(|e| format!("Error adding image to material set: {}", e))?
            .build()
            .map_err(|e| format!("Error creating material set: {}", e))?;

        Ok(Arc::new(set))
    }
}