use cgmath::Vector3;
use super::Edge;

#[derive(PartialEq, Eq, Hash)]
pub struct Face {
    centroid_index: usize,
}

impl Face {
    pub fn new<A>(edges: A) -> Self
        where A: IntoIterator
    {
        //for edge in edges
        //{
        //    if !edge.has_face() {
        //        self.add_edge(edge);
        //    }
        //
        //    edge.set_face(self);
        //}

        Face {
            centroid_index: 0,
        }
    }
}