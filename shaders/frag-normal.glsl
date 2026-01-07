#version 330 core
out vec4 FragColor;
in vec3 fNormal;
void main()
{
    // FragColor = vec4(1.0, 0.0, 0.0, 1.0);

    // trying to match
    // https://github.com/Mikubill/sd-webui-controlnet/blob/56cec5b2958edf3b1807b7e7b2b1b5186dbd2f81/annotator/midas/__init__.py
    FragColor = vec4(fNormal*0.5 + 0.5, 1.0);
};