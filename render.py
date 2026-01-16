import pyglet
import numpy as np
import ctypes
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3
from load_off import OffModel
from pyglet.window import mouse
from pyglet.window import key
from pyglet import gl

# SAM2 setup
from transformers import pipeline

sam2_mask_generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=0)

# Controlnet setup
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-normal",
    torch_dtype=torch.float16
)
controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)
controlnet_pipe.enable_xformers_memory_efficient_attention()

controlnet_pipe.enable_model_cpu_offload()

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

gl.glEnable(gl.GL_DEPTH_TEST);
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
        gl.GL_TRIANGLES,
        indices=np.array(shape.faces).flatten(),
        batch=batch,
        vertexPosition=('f', verts_np.flatten()),
        vertexColor=('f', np.ones_like(verts_np).flatten()),
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


def random_color():
    # return np.random.rand(3)
    return torch.rand((3), device="cuda")


def get_torch_depth(w, h):
    depth_data = (ctypes.c_uint32 * (w * h))()
    
    gl.glReadPixels(0, 0, w, h, gl.GL_DEPTH_COMPONENT, gl.GL_UNSIGNED_INT, depth_data)
    
    depth_array = np.frombuffer(depth_data, dtype=np.uint32).reshape(h, w)
    depth_tensor = torch.from_numpy(depth_array.astype(np.float32)).to("cuda")
    
    # 0xFFFFFFFF is the max value for a 32-bit depth capture
    depth_tensor /= float(0xFFFFFFFF)
    return depth_tensor.flip(0) # Flip to match Top-Down SAM2 coordinates

def project_to_shape(masks_list, index):
    global shape, vertex_list
    if masks_list is None or index >= len(masks_list):
        return

    # 1. Move mask to CUDA immediately to match ix and iy
    current_mask_data = masks_list[index]

    # Check if it's already a tensor, otherwise convert
    if not isinstance(current_mask_data, torch.Tensor):
        mask_tensor = torch.from_numpy(current_mask_data).to("cuda")
    else:
        mask_tensor = current_mask_data.to("cuda")

    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.squeeze(0)

    w, h = window.width, window.height
    depth_buffer = get_torch_depth(w, h)

    p_data = (ctypes.c_float * 16)(*program['p'])
    mv_data = (ctypes.c_float * 16)(*program['mv'])
    proj = torch.tensor(p_data, device="cuda").reshape(4, 4).t()
    mv = torch.tensor(mv_data, device="cuda").reshape(4, 4).t()
    mvp = proj @ mv

    verts = torch.from_numpy(np.array(shape.vertices, dtype=np.float32)).to("cuda")
    verts_h = torch.cat([verts, torch.ones((verts.shape[0], 1), device="cuda")], dim=1)
    
    clip_space = verts_h @ mvp.t()
    ndc = clip_space[:, :3] / clip_space[:, 3:4]
    
    screen_x = (ndc[:, 0] + 1) * (w / 2)
    screen_y = (1 - ndc[:, 1]) * (h / 2)
    proj_z = (ndc[:, 2] + 1) / 2.0 

    valid_mask = (screen_x >= 0) & (screen_x < w) & (screen_y >= 0) & (screen_y < h)
    
    ix = screen_x[valid_mask].long()
    iy = screen_y[valid_mask].long()
    sampled_depth = depth_buffer[iy, ix]
    
    epsilon = 0.005 
    hit_mask =( proj_z[valid_mask] <= (sampled_depth + epsilon)).to("cuda")
    visible_indices = torch.where(valid_mask)[0][hit_mask]

    if len(visible_indices) > 0:
        if not hasattr(shape, 'features'):
            shape.features = torch.zeros((len(verts), 3), device="cuda")
            shape.hits = torch.zeros((len(verts), 1), device="cuda")

        mask_values = mask_tensor[iy[hit_mask], ix[hit_mask]] # Shape: (N,)
        
        # Create a unique color based on the mask index
        torch.manual_seed(index) # Keep the color consistent for this mask index
        mask_color = torch.rand(3, device="cuda") 

        # Only update vertices where the mask is actually present
        actual_hits = mask_values > 0.5
        target_indices = visible_indices[actual_hits]

        if len(target_indices) > 0:
            shape.features[target_indices] = mask_color

        # Update VBO
        colors = shape.features.cpu().numpy().flatten()
        vertex_list.vertexColor = colors
    
    program["useNormals"] = False


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
    global controlnet_result
    controlnet_result = controlnet_pipe(
        prompt="stylish person",
        image=im,
        num_inference_steps=10,
        negative_prompt="nsfw detailed").images[0]
    Image._show(controlnet_result)
    controlnet_result.save("./images/controlnet-result.png")

def sam2_inference():
    global controlnet_result
    outputs = sam2_mask_generator(controlnet_result, points_per_batch=64)
    w, h = controlnet_result.size
    classes_vis = torch.zeros((h, w, 3), dtype=torch.float32, device="cuda")
    
    global masks
    global mask_index
    masks = outputs["masks"]
    mask_index = 0
    
    # for mask in masks:
    #     classes_vis[mask] = random_color()
    
    project_to_shape(masks, mask_index)


@window.event
def on_draw():
    window.clear()
    gl.glClearColor(0.5, 0.5, 1.0, 1.0)
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
    if symbol == key.RIGHT:
        shape_index += 1
    if symbol == key.LEFT:
        shape_index -= 1
        
    
    if symbol in (key.LEFT, key.RIGHT):
        shape_index = np.clip(shape_index, 1, 44)
        reload_shape(shape_index)
        
    if symbol == key.SPACE:
        controlnet_inference()
        sam2_inference()
    
    if symbol == key.UP:
        program["useNormals"] = True
    elif symbol == key.DOWN:
        program["useNormals"] = False
        
    global mask_index
    global masks
    # Iterate through masks
    if masks is not None:
        if symbol == key.D:
            mask_index = (mask_index + 1) % len(masks)
            print(f"Projecting Mask {mask_index}/{len(masks)}")
            project_to_shape(masks, mask_index)
            
        if symbol == key.A:
            mask_index = (mask_index - 1) % len(masks)
            print(f"Projecting Mask {mask_index}/{len(masks)}")
            project_to_shape(masks, mask_index)
        


program["useNormals"] = True
controlnet_result = None
masks = None
shape_index = 1
mask_index = 0
reload_shape(shape_index)
compute_mvp()
pyglet.app.run()