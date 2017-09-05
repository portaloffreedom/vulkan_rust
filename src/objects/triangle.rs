use cgmath::Vector3;

#[derive(Clone)]
pub struct Triangle([Vector3<f32>; 3]);

impl Triangle {
    pub fn new(a: Vector3<f32>, b: Vector3<f32>, c: Vector3<f32>) -> Self {
        Triangle([a,b,c])
    }

    pub fn a(&self) -> Vector3<f32> { self.0[0] }
    pub fn b(&self) -> Vector3<f32> { self.0[1] }
    pub fn c(&self) -> Vector3<f32> { self.0[2] }

    pub fn iter(&self) -> TriangleIter {
        TriangleIter {
            triangle: &self,
            cur: 0,
        }
    }
}

pub struct TriangleIter<'a> {
    triangle: &'a Triangle,
    cur: usize
}

impl<'a> Iterator for TriangleIter<'a> {
    type Item = Vector3<f32>;

    fn next(&mut self) -> Option<Vector3<f32>> {
        let r = match self.cur {
            0 => self.triangle.a(),
            1 => self.triangle.b(),
            2 => self.triangle.c(),
            _ => return None
        };
        self.cur += 1;
        Some(r)
    }
}
