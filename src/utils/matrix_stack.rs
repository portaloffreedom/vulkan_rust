use std::sync::Arc;
use std::collections::LinkedList;
use std::ops::Mul;

use cgmath::Matrix4;
use cgmath::SquareMatrix;

pub struct MatrixStack(LinkedList<Matrix4<f32>>);

impl MatrixStack {
    pub fn new() -> Arc<Self> {
        let mut list = LinkedList::new();
        list.push_back(Matrix4::identity());

        Arc::new(MatrixStack(list))
    }

    pub fn front(&self) -> &Matrix4<f32> {
        self.0.front().unwrap()
    }

    pub fn front_mut(&mut self) -> &mut Matrix4<f32> {
        self.0.front_mut().unwrap()
    }

    pub fn mult_matrix(&mut self, matrix: Matrix4<f32>) {
        let front: &mut Matrix4<f32> = self.front_mut();
        (*front) = front.mul(matrix);
    }


}