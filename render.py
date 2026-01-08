import pyglet
import numpy as np
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from load_off import OffModel
from pyglet.window import mouse

# Controlnet setup
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image


controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-normal",
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Pyglet setup

window = pyglet.window.Window(width=1024, height=768, resizable=True)

with open("shaders/vertex.glsl", "r") as f:
    vert_src = f.read()
with open("shaders/geometry.glsl") as f:
    geometry_src = f.read()
with open("shaders/frag-normal.glsl", "r") as f:
    frag_src = f.read()

vert_shader = Shader(vert_src, 'vertex')
geom_shader = Shader(geometry_src, 'geometry')
frag_shader = Shader(frag_src, 'fragment')
program = ShaderProgram(vert_shader, geom_shader, frag_shader)

pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST);
batch = pyglet.graphics.Batch()


shape = None
vertex_list = None
def reload_shape(shape_index):
    global shape
    global vertex_list
    if shape is not None:
        vertex_list.delete()
    
    path = f"off/{shape_index}.off"
    shape = OffModel(path)
    verts_np = np.array(shape.vertices, dtype=np.float32)
    
    vertex_list = program.vertex_list_indexed(
        len(shape.vertices), 
        pyglet.gl.GL_TRIANGLES,
        indices=np.array(shape.faces).flatten(),
        batch=batch,
        vertexPosition=('f', verts_np.flatten()),
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

    mv = view @ model
    # mvp = projection @ view @ model
    program['p'] = projection
    program['mv'] = mv


def controlnet_inference():
    # https://stackoverflow.com/questions/4986662/taking-a-screenshot-with-pyglet-fixd
    pyglet.image.get_buffer_manager().get_color_buffer().save("images/screenshot.png")
    pyglet.image.get_buffer_manager().get_depth_buffer().save("images/screenshot_depth.png")
    # https://stackoverflow.com/questions/896548/how-to-convert-a-pyglet-image-to-a-pil-image
    # pitch = -(pyglet_image.width * len('RGB'))
    # data = pyglet_image.get_data('RGB', pitch) # using the new pitch
    # im = Image.fromstring('RGB', (pyglet_image.width, pyglet_image.height), data)
    # im.show()
    im = Image.open("./images/screenshot.png").convert("RGB")
    
    # MANUALLY SWAP R AND B CHANNELS
    # r, g, b = im.split()
    # im = Image.merge("RGB", (b, g, r)) # This converts RGB to BGR layout
    
    controlnet_result = pipe(
        prompt="stylish person",
        image=im,
        num_inference_steps=20,
        negative_prompt="nsfw detailed").images[0]
    Image._show(controlnet_result)
    controlnet_result.save("./images/result.png")




@window.event
def on_draw():
    window.clear()
    pyglet.gl.glClearColor(0.5, 0.5, 1.0, 1.0)
    batch.draw()

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    print("SCROLL", x, y, scroll_x, scroll_y)
    global camera_pos
    aim = Vec3(0,0,0)
    forward = Vec3.normalize(aim - camera_pos);
    right = Vec3.normalize(Vec3.cross(forward, Vec3(0, 1, 0)));
    up = Vec3.normalize(Vec3.cross(forward, Vec3(1, 0, 0)));
    sensitivity = 0.1
    max_distance_to_object = 2.0

    # print(x,y,dx,dy)
    camera_pos += forward * scroll_y * sensitivity;
    
    # // clamp with radius=max_distance
    if camera_pos.length() > max_distance_to_object:
        camera_pos = camera_pos.normalize() * max_distance_to_object;
    
    compute_mvp()


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

@window.event
def on_key_press(symbol, modifiers):
    global shape_index
    if symbol == pyglet.window.key.RIGHT:
        shape_index += 1
    if symbol == pyglet.window.key.LEFT:
        shape_index -= 1
        
    
    if symbol in (pyglet.window.key.LEFT, pyglet.window.key.RIGHT):
        shape_index = np.clip(shape_index, 1, 44)
        reload_shape(shape_index)
        
    if symbol == pyglet.window.key.SPACE:
        controlnet_inference()



shape_index = 1
reload_shape(shape_index)
compute_mvp()
pyglet.app.run()