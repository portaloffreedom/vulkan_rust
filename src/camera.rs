use actors::ActorFrame;
use cgmath::Matrix4;

pub struct Camera {
    frame: ActorFrame,
}

impl Camera {
    pub fn new() -> Self
    {
        Camera {
            frame: ActorFrame::new(),
        }
    }

    pub fn move_forward(&mut self, delta: f32)
    {
        self.frame.move_forward(delta)
    }

    pub fn move_up(&mut self, delta: f32)
    {
        self.frame.move_up(delta)
    }

    pub fn move_right(&mut self, delta: f32)
    {
        self.frame.move_right(delta)
    }

    pub fn get_matrix(&self)
                      -> Matrix4<f32>
    {
        self.frame.get_camera_matrix(false)
    }
}