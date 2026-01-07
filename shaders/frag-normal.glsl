#version 330 core
out vec4 FragColor;
in vec3 fNormal;
void main()
{
    // FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    FragColor = vec4(fNormal, 1.0);
};