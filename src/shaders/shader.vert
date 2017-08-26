#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 tex_coords;

void main() {
    v_color = color;

    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = (position + vec2(1)) /2;
}
