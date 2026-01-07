import pyglet
import numpy as np
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from load_off import OffModel

from pyglet.window import mouse

shape = OffModel("off/1.off", 0)

window = pyglet.window.Window(width=1024, height=768, resizable=True)

with open("shaders/vertex.glsl", "r") as f:
    vert_src = f.read()

with open("shaders/geometry.glsl") as f:
    geometry_src = f.read()
    
# with open("shaders/frag-red.glsl", "r") as f:
with open("shaders/frag-normal.glsl", "r") as f:
    frag_src = f.read()

vert_shader = Shader(vert_src, 'vertex')
geom_shader = Shader(geometry_src, 'geometry')
frag_shader = Shader(frag_src, 'fragment')
# program = ShaderProgram(vert_shader, frag_shader)
program = ShaderProgram(vert_shader, geom_shader, frag_shader)

# pyglet.gl.glEnable(pyglet.gl.GL_CULL_FACE)
pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST);



batch = pyglet.graphics.Batch()
verts_np = np.array(shape.vertices, dtype=np.float32)
# verts_padded = np.insert(verts_np, 3, 1.0, axis=1) # add homogeneous

vertex_list = program.vertex_list_indexed(
    len(shape.vertices), 
    pyglet.gl.GL_TRIANGLES,
    indices=np.array(shape.faces).flatten(),
    batch=batch,
    # vertexPosition=('f', verts_padded.flatten()),
    vertexPosition=('f', verts_np.flatten()),
    # vertexColor=('f', np.array(shape.features).flatten())
)

camera_pos = Vec3(0,0,2)
def compute_mvp():
    global camera_pos
        
    verts = np.array(shape.vertices)
    v_min = verts.min(axis=0)
    v_max = verts.max(axis=0)
    center = (v_min + v_max) / 2.0
    diag = np.linalg.norm(v_max - v_min)

    model = Mat4(1.0)
    model = model.scale(Vec3(2.0 / diag, 2.0 / diag, 2.0 / diag))
    model = model.translate(Vec3(-center[0], -center[1], -center[2]))

    target = Vec3(0, 0, 0)
    up_vector = Vec3(0, 1, 0)
    view = Mat4.look_at(camera_pos, target, up_vector)

    projection = Mat4.perspective_projection(
        window.aspect_ratio,
        z_near=0.1,
        z_far=5.0,
        fov=60 # degrees
    )

    mvp = projection @ view @ model
    program['mvp'] = mvp


@window.event
def on_draw():
    window.clear()
    batch.draw()



@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global camera_pos
    aim = Vec3(0,0,0)
    forward = Vec3.normalize(aim - camera_pos);
    right = Vec3.normalize(Vec3.cross(forward, Vec3(0, 1, 0)));
    up = Vec3.normalize(Vec3.cross(forward, Vec3(1, 0, 0)));
    sensitivity = 0.01
    max_distance_to_object = 2.0

    if buttons & mouse.LEFT:
        # print(x,y,dx,dy)
        camera_pos -= dx * right * sensitivity;
        camera_pos += dy * up * sensitivity;
        
        # // clamp with radius=max_distance
        if camera_pos.length() > max_distance_to_object:
            camera_pos = camera_pos.normalize() * max_distance_to_object;
        
        
        compute_mvp()

compute_mvp()
pyglet.app.run()