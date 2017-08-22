// build.rs

use std::process::Command;
use std::env;

//TODO take a look at this: https://github.com/vulkano-rs/vulkano/tree/master/glsl-to-spirv

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let src_dir = "src/shaders";
    let glsl_validator = "glslangValidator";

    // note that there are a number of downsides to this approach, the comments
    // below detail how to improve the portability of these commands.
    for shader in ["shader.vert", "shader.frag"].iter() {

        Command::new(glsl_validator)
            .arg("-V")
            .arg(&format!("{}/{}", src_dir, shader))
            .arg("-o")
            .arg(&format!("{}/{}.spv", out_dir, shader))
            //.arg("--aml")
            .status().unwrap();

        println!("cargo:rerun-if-changed={}/{}", src_dir, shader);
        //println!("cargo:warning=compiled {} shader", shader);
    }
}
