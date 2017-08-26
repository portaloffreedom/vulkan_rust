extern crate image;

#[macro_use]
extern crate vulkano;
extern crate glfw;

//pub mod shaders;
//pub mod utils;
pub mod app;



fn main() {
    let app = app::App::new(800, 600).expect("Impossible to create Application");
    match app.run() {
        Ok(_) => {},
        Err(error_message) => {
            println!("Unexpected error running program: {}\nExiting", error_message);
            std::process::exit(1);
        },
    };

}


