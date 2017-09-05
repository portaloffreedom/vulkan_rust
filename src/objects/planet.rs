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

fn slerp(p0: &Vector3<f32>, p1: Vector3<f32>, t: f32)
         -> Vector3<f32>
{
    use cgmath::InnerSpace;

    let omega: f32 = p0.dot(p1).acos();

    let mut ret: Vector3<f32> = p0 * ((1.0_f32 - t) * omega).sin();
    ret = ret + (p1 * (t * omega).sin());
    ret = ret / (omega).sin();

    ret
}

fn subdivide_triangle(triangle: Triangle, depth: u16)
                      -> Vec<Triangle>
{
    if depth == 0 {
        return vec![triangle];
    }
    let depth = depth - 1;

    let a = triangle.a();
    let b = triangle.b();
    let c = triangle.c();

    let ab_mid = slerp(&a, b, 0.5);
    let bc_mid = slerp(&b, c, 0.5);
    let ca_mid = slerp(&c, a, 0.5);

    vec![
        subdivide_triangle(Triangle::new(a, ab_mid, ca_mid), depth),
        subdivide_triangle(Triangle::new(ab_mid, b, bc_mid), depth),
        subdivide_triangle(Triangle::new(ca_mid, bc_mid, c), depth),
        subdivide_triangle(Triangle::new(ab_mid, bc_mid, ca_mid), depth),
    ].concat()
}

pub struct Planet {
    object: Arc<Object<[Vertex]>>,
}

impl Planet {
    pub fn new(queue: Arc<Queue>, material: Arc<Material>) -> Result<Self, String>
    {
        let icosaherdon = generate_icosahedron();
        let icosahedron_sub = Planet::subdivide_icosahedron(&icosaherdon, 4);

        use cgmath::InnerSpace;
        let mut data: Vec<Vertex> = Vec::new();
        for t in icosahedron_sub {
            for v in t.iter() {
                data.push(Vertex {
                    position: v.into(),
                    normal: v.normalize().into(),
                    texture_coordinate: [0.0, 0.0],
                    color: v.into(),
                })
            }
        }

        Ok(Planet {
            object: Object::from_iter(
                queue,
                material,
                data.into_iter(),
            )?
        })
    }

    pub fn object(self)
                  -> Arc<Object<[Vertex]>>
    {
        self.object
    }

    fn subdivide_icosahedron(icosahedron: &Vec<Triangle>, depth: u16)
                             -> Vec<Triangle>
    {
        let generated_triangles: Vec<Vec<Triangle>> = icosahedron.iter()
            .map(|triangle| {
                subdivide_triangle(Triangle::new(triangle.a(), triangle.b(), triangle.c()), depth)
            })
            .collect();

        generated_triangles.concat()
    }
}