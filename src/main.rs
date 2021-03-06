extern crate image;
//extern crate vek;
extern crate cgmath;

#[macro_use]
extern crate vulkano;
extern crate glfw;
extern crate matrixstack;
extern crate rand;
//extern crate num;

pub mod shaders;
//pub mod utils;
pub mod objects;
pub mod app;
pub mod camera;
pub mod actors;


fn main() {
    let app = app::App::new(800, 600).expect("Impossible to create Application");
    match app.run() {
        Ok(_) => {}
        Err(error_message) => {
            println!("Unexpected error running program: {}\nExiting", error_message);
            std::process::exit(1);
        }
    };
}


