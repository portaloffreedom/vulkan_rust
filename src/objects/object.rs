use std::sync::Arc;

use vulkano::buffer::ImmutableBuffer;
use vulkano::buffer::BufferUsage;
use vulkano::device::Queue;

use shaders::Material;


pub struct Object<T: ? Sized> {
    material: Arc<Material>,
    vertex_buffer: Arc<ImmutableBuffer<T>>,
}

impl<T> Object<[T]> {
    pub fn from_iter<D>(queue: Arc<Queue>, material: Arc<Material>, data: D)
                        -> Result<Arc<Self>, String>
        where D: ExactSizeIterator<Item = T>,
              T: 'static + Send + Sync + Sized
    {
        let (vertex_buffer, future) = ImmutableBuffer::from_iter(
            data,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::none()
            },
            queue,
        ).map_err(|e| format!("Failed to create Vertex buffer: {}", e))?;

        // Dropping the future object will immediately block the thread until the GPU has finished
        // processing the submission

        Ok(Arc::new(Object {
            material: material,
            vertex_buffer: vertex_buffer,
        }))
    }

    pub fn vertex_buffer_ref(&self) -> &Arc<ImmutableBuffer<[T]>>
    {
        &self.vertex_buffer
    }
}