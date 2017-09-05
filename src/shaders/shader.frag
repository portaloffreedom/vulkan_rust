#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_texture_coordinates;
layout(location = 2) in vec3 v_normal;
layout(location = 3) in vec3 v_light_direction;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D tex;

void main() {
    vec3 v_normal = normalize(v_normal);
    vec3 v_light_direction = normalize(v_light_direction);

    float diff = max(0.0, dot(v_normal, v_light_direction));
    vec3 color = diff * v_color + vec3(0.1, 0.1, 0.1);

    vec3 reflection = normalize(
        reflect(-v_light_direction, v_normal)
    );

    float spec = max(0.0, dot(v_normal, reflection));

    // if diffuse light is zero, don't even bother with the pow function
    if (diff != 0) {
        float fSpec = pow(spec, 128.0);
        color += vec3(fSpec, fSpec, fSpec);
    }

    f_color = vec4(color, 1.0);
//    f_color = texture(tex, tex_coords);
//    f_color = vec4(tex_coords, 0, 1);
}
