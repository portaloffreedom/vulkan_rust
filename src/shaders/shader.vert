#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 modelview;
    mat4 modelviewproj;
    mat4 normal_matrix;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texture_coordinate;
layout(location = 3) in vec3 color;

layout(location = 0) smooth out vec3 v_color;
layout(location = 1) smooth out vec2 v_texture_coordinate;
layout(location = 2) smooth out vec3 v_normal;
layout(location = 3) smooth out vec3 v_light_direction;

out gl_PerVertex {
    vec4 gl_Position;
};

const vec3 light_position = vec3(-10.0, 10.0, 10.0);

void main()
{
    vec4 v_position4 = ubo.modelview * vec4(position, 1.0);
    vec3 v_position3 = v_position4.xyz / v_position4.w;

    v_normal = (ubo.normal_matrix * vec4(normal, 1.0)).xyz;
    v_light_direction = normalize(light_position - v_position3);

    v_color = color;

    v_texture_coordinate = texture_coordinate; // (position + vec2(1)) /2;
    gl_Position = ubo.modelviewproj * vec4(position, 1.0);
}
