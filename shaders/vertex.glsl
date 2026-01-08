#version 330 core
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexColor;
uniform mat4 mv;
uniform mat4 p;
out vec3 vPos; 
out vec3 geomColor; 

void main() {

    vec4 mv_result = mv * vec4(vertexPosition, 1.0);
    gl_Position = p * mv_result;

    // to match midas prediction for controlnet
    vPos = vec3(mv_result);
    geomColor = vertexColor; 
}