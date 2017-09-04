use cgmath::Vector3;
use super::Face;

#[derive(PartialEq, Eq, Hash)]
pub struct Edge {
    origin_index: usize,
    e_twin: Box<Edge>,
    face: Option<Face>,
}

impl Edge {
    pub fn new(origin: Vector3<f32>, destination: Vector3<f32>) -> Self {
        Edge {

        }
    }

    pub fn get_destination(&self) -> usize {
        self.e_twin.origin_index
    }

    pub fn get_face(&self) -> Option<Face> { self.face }
}