use std;
use std::sync::Arc;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineParams};
use vulkano::device::Device;
use vulkano;

use shaders::Shader;

//pub struct Pipeline {
//    vk_pipeline: Arc<GraphicsPipeline>,
//}

fn create_pipeline(device: Arc<Device>, shader: Shader) -> Arc<str> {
    let renderpass = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: vulkano::format::Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).unwrap()
    );

    Arc::new(vulkano::pipeline::GraphicsPipeline::start()
        .vertex_input(vulkano::pipeline::vertex::TwoBuffersDefinition::new())
        .vertex_shader(shader.vert_shader_module.main_entry_point(), ())
        .triangle_list()
        .viewports(std::iter::once(vulkano::pipeline::viewport::Viewport {
            origin: [0.0, 0.0],
            depth_range: 0.0 .. 1.0,
            dimensions: [800 as f32, 600 as f32],
        }))
        .fragment_shader(shader.frag_shader_module.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .render_pass(vulkano::framebuffer::Subpass::from(renderpass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
    )
}