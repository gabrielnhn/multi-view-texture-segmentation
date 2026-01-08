#version 330 core
out vec4 FragColor;
in vec3 fNormal;
in vec3 fColor;

uniform bool useNormals;

//     // https://github.com/Mikubill/sd-webui-controlnet/blob/56cec5b2958edf3b1807b7e7b2b1b5186dbd2f81/annotator/midas/__init__.py
void main() {

    if (useNormals)
    {
        vec3 n = fNormal;
        float x = n.x * 0.5 + 0.5; // Flip X: Left becomes Red
        float y = n.y * 0.5 + 0.5;  // Y is Up
        float z = n.z * 0.5 + 0.5;  // Z is Front

        z = 1 - z;
        // x = 1 - x;
        y = 1 - y;
        FragColor = vec4(x,y,z, 1.0);
    }
    else
    {
        FragColor = vec4(fColor, 1);
    }
}
