import pyglet
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3

from load_off import OffModel
import numpy as np

shape = OffModel("off/1.off", 0)

window = pyglet.window.Window(width=1024, height=768, resizable=True)

with open("shaders/vertex.glsl", "r") as f:
    vert_src = f.read()
with open("shaders/frag-red.glsl", "r") as f:
    frag_src = f.read()

vert_shader = Shader(vert_src, 'vertex')
frag_shader = Shader(frag_src, 'fragment')
program = ShaderProgram(vert_shader, frag_shader)

batch = pyglet.graphics.Batch()
verts_np = np.array(shape.vertices, dtype=np.float32)
verts_padded = np.insert(verts_np, 3, 1.0, axis=1) # add homogeneous

vertex_list = program.vertex_list_indexed(
    len(shape.vertices), 
    pyglet.gl.GL_TRIANGLES,
    indices=np.array(shape.faces).flatten(),
    batch=batch,
    vertexPosition=('f', verts_padded.flatten()),
    vertexColor=('f', np.array(shape.features).flatten())
)


verts = np.array(shape.vertices)
v_min = verts.min(axis=0)
v_max = verts.max(axis=0)

# v_max = np.array([float('-inf'), float('-inf'), float('-inf')])
# v_min = np.array([float('inf'), float('inf'), float('inf')])
# for vert in verts:
#     for idx in range(0,3):
#         if vert[idx] < v_min[idx]:
#             v_min[idx] = vert[idx]
            
#         if vert[idx] > v_max[idx]:
#             v_max[idx] = vert[idx]
# print("v_min is ", v_min)
# print("v_max is ", v_max)

center = (v_min + v_max) / 2.0
diag = np.linalg.norm(v_max - v_min)

model = Mat4(1.0)
model = model.scale(Vec3(2.0 / diag, 2.0 / diag, 2.0 / diag))
model = model.translate(Vec3(-center[0], -center[1], -center[2]))

# Equivalent to glm::lookAt(camera_pos, target, up)
camera_pos = Vec3(0, 0, 2) # Example position
target = Vec3(0, 0, 0)
up_vector = Vec3(0, 1, 0)
view = Mat4.look_at(camera_pos, target, up_vector)

import math
# fov = math.radians(60)
projection = Mat4.perspective_projection(
    window.aspect_ratio,
    z_near=0.1,
    z_far=5.0,
    fov=60
)

mvp = projection @ view @ model
program['mvp'] = mvp

@window.event
def on_draw():
    window.clear()
    batch.draw()

pyglet.app.run()