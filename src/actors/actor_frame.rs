use cgmath::{Vector3, Vector4};
use cgmath::Matrix4;
use cgmath::Matrix;
use cgmath::Zero;

pub struct ActorFrame {
    location: Vector3<f32>,
    up: Vector3<f32>,
    forward: Vector3<f32>,
}

impl ActorFrame {
    pub fn new() -> Self
    {
        ActorFrame {
            location: Vector3::zero(),
            up: Vector3 { x: 0.0, y: 1.0, z: 0.0 },
            forward: Vector3 { x: 0.0, y: 0.0, z: -1.0 },
        }
    }

    pub fn get_matrix(&self, rotation_only: bool)
                      -> Matrix4<f32>
    {
        get_matrix(self.location, self.up, self.forward, rotation_only)
    }

    pub fn get_camera_matrix(&self, rotation_only: bool)
                             -> Matrix4<f32>
    {
        let matrix = get_matrix(self.location, self.up, -self.forward, true)
            .transpose();

        if rotation_only {
            matrix
        } else {
            let translation_matrix = Matrix4::from_translation(-self.location);

            matrix * translation_matrix
        }
    }

    pub fn move_forward(&mut self, delta: f32)
    {
        self.location += self.forward * delta;
    }

    pub fn move_up(&mut self, delta: f32)
    {
        self.location += self.up * delta;
    }

    pub fn move_right(&mut self, delta: f32)
    {
        let cross = self.up.cross(self.forward);
        self.location += cross * delta;
    }
}


fn get_matrix(location: Vector3<f32>, up: Vector3<f32>, forward: Vector3<f32>, rotation_only: bool)
              -> Matrix4<f32>
{
    let x_axis = up.cross(forward);

    let last_column: Vector4<f32> = if rotation_only {
        Vector4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    } else {
        location.extend(1.0)
    };

    Matrix4::from_cols(
        x_axis.extend(0.0),
        up.extend(0.0),
        forward.extend(0.0),
        last_column,
    )
}