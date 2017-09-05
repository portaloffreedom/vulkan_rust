use std::f32;
use std::sync::Arc;

use vulkano::device::Queue;

use objects::Triangle;
use cgmath::Vector3;
use objects::Object;
use shaders::Vertex;
use shaders::Material;




fn generate_icosahedron() -> Vec<Triangle>
{
    let phi: f32 = (1.0_f32 + 5.0_f32.sqrt()) / 2.0_f32;
    let du: f32 = 1.0_f32 / (phi * phi + 1.0_f32).sqrt();
    let dv: f32 = phi * du;

    vec![
        Triangle::new(Vector3{ x:0.0, y: dv, z: du }, Vector3{ x:0.0, y: dv, z:-du }, Vector3{ x: dv, y: du, z:0.0 }), // 0 1 8
        Triangle::new(Vector3{ x:0.0, y: dv, z: du }, Vector3{ x: du, y:0.0, z: dv }, Vector3{ x:-du, y:0.0, z: dv }), // 0 4 5
        Triangle::new(Vector3{ x:0.0, y: dv, z: du }, Vector3{ x:-du, y:0.0, z: dv }, Vector3{ x:-dv, y: du, z:0.0 }), // 0 5 10
        Triangle::new(Vector3{ x:0.0, y: dv, z: du }, Vector3{ x: dv, y: du, z:0.0 }, Vector3{ x: du, y:0.0, z: dv }), // 0 8 4
        Triangle::new(Vector3{ x:0.0, y: dv, z: du }, Vector3{ x:-dv, y: du, z:0.0 }, Vector3{ x:0.0, y: dv, z:-du }), // 0 10 1
        Triangle::new(Vector3{ x:0.0, y: dv, z:-du }, Vector3{ x: du, y:0.0, z:-dv }, Vector3{ x: dv, y: du, z:0.0 }), // 1 6 8
        Triangle::new(Vector3{ x:0.0, y: dv, z:-du }, Vector3{ x:-du, y:0.0, z:-dv }, Vector3{ x: du, y:0.0, z:-dv }), // 1 7 6
        Triangle::new(Vector3{ x:0.0, y: dv, z:-du }, Vector3{ x:-dv, y: du, z:0.0 }, Vector3{ x:-du, y:0.0, z:-dv }), // 1 10 7
        Triangle::new(Vector3{ x:0.0, y:-dv, z: du }, Vector3{ x:0.0, y:-dv, z:-du }, Vector3{ x:-dv, y:-du, z:0.0 }), // 2 3 11
        Triangle::new(Vector3{ x:0.0, y:-dv, z: du }, Vector3{ x: du, y:0.0, z: dv }, Vector3{ x: dv, y:-du, z:0.0 }), // 2 4 9
        Triangle::new(Vector3{ x:0.0, y:-dv, z: du }, Vector3{ x:-du, y:0.0, z: dv }, Vector3{ x: du, y:0.0, z: dv }), // 2 5 4
        Triangle::new(Vector3{ x:0.0, y:-dv, z: du }, Vector3{ x: dv, y:-du, z:0.0 }, Vector3{ x:0.0, y:-dv, z:-du }), // 2 9 3
        Triangle::new(Vector3{ x:0.0, y:-dv, z: du }, Vector3{ x:-dv, y:-du, z:0.0 }, Vector3{ x:-du, y:0.0, z: dv }), // 2 11 5
        Triangle::new(Vector3{ x:0.0, y:-dv, z:-du }, Vector3{ x:-du, y:0.0, z:-dv }, Vector3{ x:-dv, y:-du, z:0.0 }), // 3 7 11
        Triangle::new(Vector3{ x:0.0, y:-dv, z:-du }, Vector3{ x: du, y:0.0, z:-dv }, Vector3{ x:-du, y:0.0, z:-dv }), // 3 6 7
        Triangle::new(Vector3{ x:0.0, y:-dv, z:-du }, Vector3{ x: dv, y:-du, z:0.0 }, Vector3{ x: du, y:0.0, z:-dv }), // 3 9 6
        Triangle::new(Vector3{ x: du, y:0.0, z: dv }, Vector3{ x: dv, y: du, z:0.0 }, Vector3{ x: dv, y:-du, z:0.0 }), // 4 8 9
        Triangle::new(Vector3{ x:-du, y:0.0, z: dv }, Vector3{ x:-dv, y:-du, z:0.0 }, Vector3{ x:-dv, y: du, z:0.0 }), // 5 11 10
        Triangle::new(Vector3{ x: du, y:0.0, z:-dv }, Vector3{ x: dv, y:-du, z:0.0 }, Vector3{ x: dv, y: du, z:0.0 }), // 6 9 8
        Triangle::new(Vector3{ x:-du, y:0.0, z:-dv }, Vector3{ x:-dv, y: du, z:0.0 }, Vector3{ x:-dv, y:-du, z:0.0 }), // 7 10 11
    ]
}

pub struct Planet {
    object: Arc<Object<[Vertex]>>,
}

impl Planet {
    pub fn new(queue: Arc<Queue>, material: Arc<Material>) -> Result<Self, String>
    {
        let icosaherdon = generate_icosahedron();
//        let icosahedron_sub = Planet::subdivide_icosahedron(&icosaherdon);

        use cgmath::InnerSpace;
        let mut data: Vec<Vertex> = Vec::new();
        for t in icosaherdon {
            for v in t.iter() {
                data.push(Vertex {
                    position: v.into(),
                    normal: v.normalize().into(),
                    texture_coordinate: [0.0, 0.0],
                    color: v.into(),
                })
            }
        }

        Ok(Planet{
            object: Object::from_iter(
                queue,
                material,
                data.into_iter(),
            )?
        })
    }

    pub fn object(self) -> Arc<Object<[Vertex]>>
    {
        self.object
    }

//    fn generate_icosahedron() -> Polygon {
//
//        let mut icosahedron = Polygon::new();
//
//        for triangle in ICOSAHEDRON.into_iter() {
//            icosahedron.add_triangle(triangle);
//        }
//
//        icosahedron
//    }
//
//    fn subdivide_icosahedron(icosahedron: &Polygon) {
//
//    }
}