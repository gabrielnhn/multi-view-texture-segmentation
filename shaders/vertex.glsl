#version 330 core
layout (location = 0) in vec3 vertexPosition;
uniform mat4 mvp;
out vec3 vPos; 

void main() {
    gl_Position = mvp * vec4(vertexPosition, 1.0);
    vPos = vec3(gl_Position); 
}