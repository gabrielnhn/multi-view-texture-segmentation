#version 330 core
layout (location = 0) in vec4 vertexPosition;
layout (location = 1) in vec3 vertexColor;
uniform mat4 mvp;
out vec4 geomPos;
out vec3 geomColor;
// out float vertexShade;
void main()
{
    // vertexShade = vertexPosition.z;  // Pass the position directly to the fragment shader for color
    gl_Position = mvp * vertexPosition;  // Apply MVP transformation
    geomPos = vertexPosition;
    geomColor = vertexColor;
};

