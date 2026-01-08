#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 vPos[];
out vec3 fNormal;

in vec3 geomColor[];
out vec3 fColor;

// uniform vec3 cameraPos;


void main() {
    fColor = geomColor[0];


    vec3 edge1 = vPos[1] - vPos[0];
    vec3 edge2 = vPos[2] - vPos[0];
    // vec3 normal = normalize(cross(edge1, edge2));
    vec3 normal = cross(edge1, edge2);
    // normal.x /= (2.0 * 3.14159);
    // normal.y /= (2.0 * 3.14159);
    normal = normalize(normal);



    vec3 viewDirection = vPos[0];

    if (dot(normal, viewDirection) <= 0)
        normal = - normal;

    for(int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        fNormal = normal; // Same normal for all 3 vertices
        EmitVertex();
    }
    EndPrimitive();
}