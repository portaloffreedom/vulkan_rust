mod shader;
mod material;
//mod pipeline;

pub use self::shader::Shader;
pub use self::material::Material;
//pub use pipeline::Pipeline;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub texture_coordinate: [f32; 2],
    pub color: [f32; 3],
}

impl_vertex!(Vertex, position, texture_coordinate, color);