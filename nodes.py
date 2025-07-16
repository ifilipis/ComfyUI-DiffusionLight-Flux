import os
import torch
import numpy as np
import torch.nn.functional as F
from .relighting.tonemapper import TonemapHDR
from HDRutils.exposures import estimate_exposures
import folder_paths

def create_envmap_grid(size: int):
    """
    BLENDER CONVENSION
    Create the grid of environment map that contain the position in sperical coordinate
    Top left is (0,0) and bottom right is (pi/2, 2pi)
    """    
    theta = torch.linspace(0, np.pi * 2, size * 2)
    phi = torch.linspace(0, np.pi, size)
    
    #use indexing 'xy' torch match vision's homework 3
    theta, phi = torch.meshgrid(theta, phi ,indexing='xy') 
    
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
    theta_phi = theta_phi.numpy()
    return theta_phi

def get_normal_vector(incoming_vector: np.ndarray, reflect_vector: np.ndarray):
    """
    BLENDER CONVENSION
    incoming_vector: the vector from the point to the camera
    reflect_vector: the vector from the point to the light source
    """
    #N = 2(R ⋅ I)R - I
    N = (incoming_vector + reflect_vector) / np.linalg.norm(incoming_vector + reflect_vector, axis=-1, keepdims=True)
    return N

def get_cartesian_from_spherical(theta: np.array, phi: np.array, r = 1.0):
    """
    BLENDER CONVENSION
    theta: vertical angle
    phi: horizontal angle
    r: radius
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.concatenate([x[...,None],y[...,None],z[...,None]], axis=-1)

class chrome_ball_to_envmap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ball_images": ("IMAGE", ),
                "envmap_height": ("INT", {"default": 256, "min": 1, "max": 2048, "step": 1}, ),
                "scale": ("INT", {"default": 4, "min": 1, "max": 30, "step": 1}, ),
            },
        }
        
    CATEGORY = "DiffusionLight"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",  )

    FUNCTION = "process"

   
    def process(self, ball_images, envmap_height, scale):     
        I = np.array([1, 0, 0])

        # compute  normal map that create from reflect vector
        env_grid = create_envmap_grid(envmap_height * scale)   
        reflect_vec = get_cartesian_from_spherical(env_grid[...,1], env_grid[...,0])
        normal = get_normal_vector(I[None,None], reflect_vec)
        
        # turn from normal map to position to lookup [Range: 0,1]
        pos = (normal + 1.0) / 2
        pos  = 1.0 - pos
        pos = pos[...,1:]
        
        env_map = None
        # convert position to pytorch grid look up
        grid = torch.from_numpy(pos)[None].float()
        grid = grid * 2 - 1 # convert to range [-1,1]
        print(grid.shape)
        ball_images = ball_images.permute(0,3,1,2) # [1,3,H,W]

        env_maps_list = []
        for ball in ball_images:
            env_map = F.grid_sample(ball.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True)
            env_map_default = F.interpolate(env_map, size=(envmap_height, envmap_height*2), mode='bilinear', align_corners=True)
            env_map_default = env_map_default.permute(0,2,3,1).cpu().to(torch.float32)
            env_maps_list.append(env_map_default)
        env_maps_out = torch.cat(env_maps_list, dim=0)

        
 
        return env_maps_out,

class exposure_to_hdr:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                #"EV": ("FLOAT", {"default": 0, "min": 1, "max": 30, "step": 1}, ),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1, "max": 30, "step": 0.01}, ),
            },
        }
        
    CATEGORY = "DiffusionLight"
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("hdr_image", "ldr_image", )

    FUNCTION = "exposuretohdr"

    def exposuretohdr(self, images, gamma):
        NOISE = 4
        SAT_F = 1.0
        images_np = images.detach().cpu().numpy()
        dtype = np.uint16
        dtype_max = np.iinfo(dtype).max
        imgs_uint = np.clip(images_np * dtype_max, 0, dtype_max).astype(dtype)

        H, W, _ = imgs_uint[0].shape

        luma = (
            0.2126 * imgs_uint[..., 0] +
            0.7152 * imgs_uint[..., 1] +
            0.0722 * imgs_uint[..., 2]
        ).astype(dtype)

        metadata = {
            'black_level':      np.zeros(4, dtype=int),
            'saturation_point': dtype_max * SAT_F,
            'dtype':            dtype,
            'h':                H,
            'w':                W
        }

        dummy = np.ones(len(imgs_uint), dtype=np.float32)
        exp = estimate_exposures(luma, dummy, metadata, 'mst', noise_floor=NOISE)

        imgs_lin = imgs_uint.astype(np.float32) / dtype_max

        def srgb_to_linear(c):
            mask = c <= 0.04045
            return np.where(mask, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

        rad = np.zeros_like(imgs_lin[0], dtype=np.float32)
        wgt = np.zeros_like(imgs_lin[0], dtype=np.float32)

        for im, t in zip(imgs_lin, exp):
            valid = (im > (NOISE / dtype_max)) & (im < SAT_F)
            w = np.where(im <= 0.5, im, 1.0 - im)
            w *= valid.astype(np.float32)
            rad += w * (im / t)
            wgt += w

        hdr = rad / (wgt + 1e-6)
        hdr /= hdr.max()
        hdr = srgb_to_linear(hdr)

        hdr_rgb = torch.from_numpy(hdr)
        hdr2ldr = TonemapHDR(gamma=gamma, percentile=99, max_mapping=0.9)
        ldr_rgb, _, _ = hdr2ldr(hdr_rgb)

        hdr_rgb = hdr_rgb.unsqueeze(0).cpu().to(torch.float32)
        ldr_rgb = ldr_rgb.unsqueeze(0).cpu().to(torch.float32)

        return (hdr_rgb, ldr_rgb,)


class SaveImageOpenEXR:
    def __init__(self):
        try:
            import OpenEXR
            import Imath
            self.OpenEXR = OpenEXR
            self.Imath = Imath
            self.use_openexr = True
        except ImportError:
            print("No OpenEXR module found, trying OpenCV...")
            self.use_openexr = False
            try:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                import cv2
                self.cv2 = cv2
            except ImportError:
                raise ImportError("No OpenEXR or OpenCV module found, can't save EXR")

        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_EXR"})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_url",)
    FUNCTION = "saveexr"
    OUTPUT_NODE = True
    CATEGORY = "DiffusionLight"

    def saveexr(self, images, filename_prefix):
        import re
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        def file_counter():
            max_counter = 0
            for existing_file in os.listdir(full_output_folder):
                match = re.fullmatch(rf"{filename}_(\d+)_?\.[a-zA-Z0-9]+", existing_file)
                if match:
                    fc = int(match.group(1))
                    if fc > max_counter:
                        max_counter = fc
            return max_counter

        for image in images:
            image_np = image.cpu().numpy().astype(np.float32)

            if self.use_openexr:
                PIXEL_TYPE = self.Imath.PixelType(self.Imath.PixelType.FLOAT)
                height, width, channels = image_np.shape
                header = self.OpenEXR.Header(width, height)
                half_chan = self.Imath.Channel(PIXEL_TYPE)
                header['channels'] = dict([(c, half_chan) for c in "RGB"])
                R = image_np[:, :, 0].tobytes()
                G = image_np[:, :, 1].tobytes()
                B = image_np[:, :, 2].tobytes()
                counter = file_counter() + 1
                file = f"{filename}_{counter:05}.exr"
                exr_file = self.OpenEXR.OutputFile(os.path.join(full_output_folder, file), header)
                exr_file.writePixels({'R': R, 'G': G, 'B': B})
                exr_file.close()
            else:
                counter = file_counter() + 1
                file = f"{filename}_{counter:05}.exr"
                exr_file = os.path.join(full_output_folder, file)
                self.cv2.imwrite(exr_file, image_np)

        return (f"/view?filename={file}&subfolder=&type=output",)

NODE_CLASS_MAPPINGS = {
    "chrome_ball_to_envmap": chrome_ball_to_envmap,
    "exposure_to_hdr": exposure_to_hdr,
    "SaveImageOpenEXR": SaveImageOpenEXR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "chrome_ball_to_envmap": "Chrome Ball to Envmap",
    "exposure_to_hdr": "Exposure to HDR",
    "SaveImageOpenEXR": "Save Image as OpenEXR",
}
