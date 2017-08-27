#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D tex;

void main() {
    f_color = vec4(v_color, 1.0);
    f_color = texture(tex, tex_coords);
//    f_color = vec4(tex_coords, 0, 1);
}
