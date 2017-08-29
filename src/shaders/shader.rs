use std::borrow::Cow;
use std::path::Path;
use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::result::Result;

use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::descriptor::descriptor::DescriptorDesc;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use vulkano::device::Device;
use vulkano::format;
use vulkano::pipeline::shader::GraphicsShaderType;
use vulkano::pipeline::shader::GraphicsEntryPoint;
use vulkano::pipeline::shader::ShaderModule;
use vulkano::pipeline::shader::{ShaderInterfaceDef, ShaderInterfaceDefEntry};

pub struct Shader {
    device: Arc<Device>,
    vert_shader_module: Arc<ShaderModule>,
    frag_shader_module: Arc<ShaderModule>,
}

impl Shader {
    pub fn new<P: AsRef<Path>>(device: Arc<Device>, vert_shader_filepath: P, frag_shader_file_path: P) -> Result<Arc<Shader>, String> {
        let vert_shader_module = create_shader_module(device.clone(), vert_shader_filepath)?;
        let frag_shader_module = create_shader_module(device.clone(), frag_shader_file_path)?;

        Ok(Arc::new(Shader {
            device: device,
            vert_shader_module: vert_shader_module,
            frag_shader_module: frag_shader_module,
        }))
    }

    pub fn entry_points(&self)
                        -> Result<(GraphicsEntryPoint<(), VertInput, VertOutput, VertLayout>,
                                   GraphicsEntryPoint<(), FragInput, FragOutput, FragLayout>), String>
    {
        // NOTE: ShaderModule::*_shader_entry_point calls do not do any error
        // checking and you have to verify correctness of what you are doing by
        // yourself.
        //
        // You must be extra careful to specify correct entry point, or program will
        // crash at runtime outside of rust and you will get NO meaningful error
        // information!
        let vert_main = unsafe {
            self.vert_shader_module.graphics_entry_point(
                CStr::from_bytes_with_nul_unchecked(b"main\0"),
                VertInput,
                VertOutput,
                VertLayout(ShaderStages { vertex: true, ..ShaderStages::none() }),
                GraphicsShaderType::Vertex
            )
        };

        let frag_main = unsafe {
            self.frag_shader_module.graphics_entry_point(
                CStr::from_bytes_with_nul_unchecked(b"main\0"),
                FragInput,
                FragOutput,
                FragLayout(ShaderStages { fragment: true, ..ShaderStages::none() }),
                GraphicsShaderType::Fragment
            )
        };

        Ok((vert_main, frag_main))
    }
}

fn create_shader_module<P: AsRef<Path>>(device: Arc<Device>, file_path: P)
                                        -> Result<Arc<ShaderModule>, String>
{
    let mut f = File::open(&file_path).map_err(|e| format!("Error loading shader: {} - Error: {}", file_path.as_ref().display(), e))?;
    let mut v = vec![];
    f.read_to_end(&mut v).map_err(|e| format!("Impossible to read vertex shader file: {}", e))?;
    // Create a ShaderModule on a device the same Shader::load does it.
    // NOTE: You will have to verify correctness of the data by yourself!
    unsafe { ShaderModule::new(device, &v) }.map_err(|e| format!("Impossible to create shader module: {}", e))
}

// This structure will tell Vulkan how input entries of our vertex shader
// look like.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VertInput;

unsafe impl ShaderInterfaceDef for VertInput {
    type Iter = VertInputIter;

    fn elements(&self) -> VertInputIter {
        VertInputIter(0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct VertInputIter(u16);

impl Iterator for VertInputIter {
    type Item = ShaderInterfaceDefEntry;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // There are things to consider when giving out entries:
        // * There must be only one entry per one location, you can't have
        //   `color' and `position' entries both at 0..1 locations.  They also
        //   should not overlap.
        // * Format of each element must be no larger than 128 bits.
        if self.0 == 0 {
            self.0 += 1;
            Some(ShaderInterfaceDefEntry {
                location: 0..1,
                format: format::Format::R32G32Sfloat,
                name: Some(Cow::Borrowed("position"))
            })
        } else if self.0 == 1 {
            self.0 += 1;
            Some(ShaderInterfaceDefEntry {
                location: 1..2,
                format: format::Format::R32G32Sfloat,
                name: Some(Cow::Borrowed("texture_coordinate"))
            })
        } else if self.0 == 2 {
            self.0 += 1;
            Some(ShaderInterfaceDefEntry {
                location: 2..3,
                format: format::Format::R32G32B32Sfloat,
                name: Some(Cow::Borrowed("color"))
            })
        } else {
            None
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // We must return exact number of entries left in iterator.
        let len = (2 - self.0) as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for VertInputIter {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VertOutput;

unsafe impl ShaderInterfaceDef for VertOutput {
    type Iter = VertOutputIter;

    fn elements(&self) -> VertOutputIter {
        VertOutputIter(0)
    }
}
// This structure will tell Vulkan how output entries (those passed to next
// stage) of our vertex shader look like.
#[derive(Debug, Copy, Clone)]
pub struct VertOutputIter(u16);

impl Iterator for VertOutputIter {
    type Item = ShaderInterfaceDefEntry;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            self.0 += 1;
            Some(ShaderInterfaceDefEntry {
                location: 0..1,
                format: format::Format::R32G32B32Sfloat,
                name: Some(Cow::Borrowed("v_color"))
            })
        } else if self.0 == 1 {
            self.0 += 1;
            Some(ShaderInterfaceDefEntry {
                location: 1..2,
                format: format::Format::R32G32Sfloat,
                name: Some(Cow::Borrowed("tex_coords"))
            })
        } else {
            None
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (1 - self.0) as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for VertOutputIter {}
// This structure describes layout of this stage.
#[derive(Debug, Copy, Clone)]
pub struct VertLayout(ShaderStages);

unsafe impl PipelineLayoutDesc for VertLayout {
    // Number of descriptor sets it takes.
    fn num_sets(&self) -> usize { 2 }
    // Number of entries (bindings) in each set.
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set {
            0 => Some(1),
            1 => Some(1),
            _ => None,
        }
    }
    // Descriptor descriptions.
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        use vulkano::descriptor::descriptor::{DescriptorDescTy, ShaderStages, DescriptorImageDesc, DescriptorImageDescDimensions, DescriptorImageDescArray, DescriptorBufferDesc};
        use vulkano::format::Format;

        match (set, binding) {
            (0, 0) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(
                    DescriptorBufferDesc {
                        dynamic: Some(false),
                        storage: false,
                    }
                ),
                array_count: 1,
                stages: ShaderStages {
                    vertex: true,
                    .. ShaderStages::none()
                },
                readonly: true,
            }),
            _ => None,
        }
    }
    // Number of push constants ranges (think: number of push constants).
    fn num_push_constants_ranges(&self) -> usize { 0 }
    // Each push constant range in memory.
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        if num != 0 || 0 == 0 { return None; }
        Some(PipelineLayoutDescPcRange {
            offset: 0,
            size: 0,
            stages: ShaderStages::all()
        })
    }
}

// Same as with our vertex shader, but for fragment one instead.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FragInput;

unsafe impl ShaderInterfaceDef for FragInput {
    type Iter = FragInputIter;

    fn elements(&self) -> FragInputIter {
        FragInputIter(0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FragInputIter(u16);

impl Iterator for FragInputIter {
    type Item = ShaderInterfaceDefEntry;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            self.0 += 1;
            Some(ShaderInterfaceDefEntry {
                location: 0..1,
                format: format::Format::R32G32B32Sfloat,
                name: Some(Cow::Borrowed("v_color"))
            })
        } else if self.0 == 1 {
            self.0 += 1;
            Some(ShaderInterfaceDefEntry {
                location: 1..2,
                format: format::Format::R32G32Sfloat,
                name: Some(Cow::Borrowed("tex_coords"))
            })
        } else {
            None
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (1 - self.0) as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for FragInputIter {}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FragOutput;

unsafe impl ShaderInterfaceDef for FragOutput {
    type Iter = FragOutputIter;

    fn elements(&self) -> FragOutputIter {
        FragOutputIter(0)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FragOutputIter(u16);

impl Iterator for FragOutputIter {
    type Item = ShaderInterfaceDefEntry;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Note that color fragment color entry will be determined
        // automatically by Vulkano.
        if self.0 == 0 {
            self.0 += 1;
            return Some(ShaderInterfaceDefEntry {
                location: 0..1,
                format: format::Format::R32G32B32A32Sfloat,
                name: Some(Cow::Borrowed("f_color"))
            });
        }
        None
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (1 - self.0) as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for FragOutputIter {}
// Layout same as with vertex shader.
#[derive(Debug, Copy, Clone)]
pub struct FragLayout(ShaderStages);

unsafe impl PipelineLayoutDesc for FragLayout {
    fn num_sets(&self) -> usize { 2 }
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set {
            0 => Some(1),
            1 => Some(1),
            _ => None,
        }
    }
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        use vulkano::descriptor::descriptor::{DescriptorDescTy, ShaderStages, DescriptorImageDesc, DescriptorImageDescDimensions, DescriptorImageDescArray, DescriptorBufferDesc};
        use vulkano::format::Format;

        match (set, binding) {
            (1, 0) => Some(DescriptorDesc {
                ty: DescriptorDescTy::CombinedImageSampler({
                    DescriptorImageDesc {
                        sampled: true,
                        dimensions: DescriptorImageDescDimensions::TwoDimensional,
                        format: Some(Format::R8G8B8A8Srgb),
                        multisampled: false,
                        array_layers: DescriptorImageDescArray::NonArrayed,
                    }
                }),
                array_count: 1,
                stages: ShaderStages {
                    fragment: true,
                    .. ShaderStages::none()
                },
                readonly: true,
            }),
            _ => None,
        }
    }
    fn num_push_constants_ranges(&self) -> usize { 0 }
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        if num != 0 || 0 == 0 { return None; }
        Some(PipelineLayoutDescPcRange {
            offset: 0,
            size: 0,
            stages: ShaderStages::all()
        })
    }
}