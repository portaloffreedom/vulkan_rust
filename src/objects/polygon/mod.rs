pub mod edge;
pub mod face;
pub mod tile;

use self::edge::Edge;
use self::face::Face;
use objects::Triangle;

use std::collections::HashSet;
use cgmath::Vector3;

pub struct Polygon {
    vertices: Vec<Vector3<f32>>,
    edges: HashSet<Edge>,
    faces: HashSet<Face>,
}

impl Polygon {
    pub fn new() -> Self {
        Polygon {
            vertices: Vec::new(),
            edges: HashSet::new(),
            faces: HashSet::new(),
        }
    }

    pub fn add_triangle(&mut self, triangle: Triangle) -> Face {
        let ab = self.add_edge(triangle.a(), triangle.b());

        // test if this triangle already exists
        if let Some(face) = ab.get_face() {
            return face
        }

        let bc = self.add_edge(triangle.b(), triangle.c());
        let ca = self.add_edge(triangle.c(), triangle.a());
        let face = Face::new([ab, bc, ca].iter());

        self.faces.insert(face);
        return face;
    }

    pub fn add_edge<'a>(&mut self, origin: Vector3<f32>, destination: Vector3<f32>) -> &'a Edge {
        let o_index = self.add_vertex(origin);
        let d_index = self.add_vertex(destination);

        for edge in &self.edges {
            if edge.get_destination() == d_index {
                return edge;
            }
        }

        let edge = Edge::new(origin, destination);
        self.edges.insert(edge);
        return self.edges.get(&edge).unwrap();
    }

    pub fn add_vertex(&mut self, vertex: Vector3<f32>) -> usize {
        for (index, v) in self.vertices.into_iter().enumerate() {
            if v == vertex {
                return index;
            }
        }

        self.vertices.push(vertex);
        self.vertices.len()-1
    }
}
