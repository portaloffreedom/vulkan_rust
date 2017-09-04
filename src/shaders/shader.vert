#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 modelview;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 tex_coords;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    v_color = color;

    gl_Position = ubo.proj * ubo.modelview * vec4(position, 1.0);
    tex_coords = texture_coordinate; // (position + vec2(1)) /2;
}
