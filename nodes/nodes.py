from comfy.utils import common_upscale
import torch
import comfy.model_management as mm
import comfy.ldm.wan.vae
import node_helpers
import nodes

original_encode = comfy.ldm.wan.vae.WanVAE.encode
original_decode = comfy.ldm.wan.vae.WanVAE.decode

# reference: https://github.com/raindrop313/ComfyUI-WanVideoStartEndFrames
# reference: https://github.com/kijai/ComfyUI-WanVideoWrapper/
def encode(self, x):
    self.clear_cache()

    t = x.shape[2]
    iter_ = 2 + (t - 2) // 4

    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx)
        elif i== iter_-1:
            out_ = self.encoder(x[:, :, -1:, :, :],
                                feat_cache=[None] * self._enc_conv_num,
                                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
        else:
            out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
    out_head = out[:, :, :iter_ - 1, :, :]
    out_tail = out[:, :, -1, :, :].unsqueeze(2)
    mu, log_var = torch.cat([self.conv1(out_head), self.conv1(out_tail)], dim=2).chunk(2, dim=1)
    return mu

def decode(self, z):
    self.clear_cache()
    
    iter_ = z.shape[2]
    z_head=z[:,:,:-1,:,:]
    z_tail=z[:,:,-1,:,:].unsqueeze(2)
    x = torch.cat([self.conv2(z_head), self.conv2(z_tail)], dim=2)
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:, :, i:i + 1, :, :],
                                feat_cache=self._feat_map,
                                feat_idx=self._conv_idx)
        elif i==iter_-1:
            out_ = self.decoder(x[:, :, -1, :, :].unsqueeze(2),
                                feat_cache=None,
                                feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
        else:
            out_ = self.decoder(x[:, :, i:i + 1, :, :],
                                feat_cache=self._feat_map,
                                feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    return out
    
class WanImageToVideo_F2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 320, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 320, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": 1000, "step": 4}),
                             "start_image": ("IMAGE", ),
                },
                "optional": {"end_image": ("IMAGE", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "initialize"

    CATEGORY = "Flow2/Wan 2.1"

    def initialize(self, positive, negative, vae, width, height, length, start_image=None, end_image=None, clip_vision_output=None):
        valid_end_image = end_image is not None

        if valid_end_image:
            comfy.ldm.wan.vae.WanVAE.encode = encode
            comfy.ldm.wan.vae.WanVAE.decode = decode
        else:
            comfy.ldm.wan.vae.WanVAE.encode = original_encode
            comfy.ldm.wan.vae.WanVAE.decode = original_decode

        offload_device = mm.unet_offload_device()
        device = offload_device
        
        resized_start_image = common_upscale(start_image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if valid_end_image:
            resized_end_image = common_upscale(end_image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        empty_frames = torch.ones(length - 1, height, width, 3, device=device) * 0.5

        if valid_end_image:
            concatenated = torch.cat([resized_start_image.to(device), empty_frames, resized_end_image.to(device)], dim=0)
        else:
            concatenated = torch.cat([resized_start_image.to(device), empty_frames], dim=0)
            
        la = vae.encode(concatenated)
    
        latent = torch.zeros([1, 16, ((length - 1) // 4) + 1 + int(valid_end_image), height // 8, width // 8], device=device)
        
        mask = torch.ones((1, length + int(valid_end_image), la.shape[-2], la.shape[-1]), device=device)
        mask[:, 0] = 0.0
        if valid_end_image:
            mask[:, -1] = 0.0

        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1)

        mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, la.shape[-2], la.shape[-1])
        mask = mask.movedim(1, 2)
        mask = mask.repeat(1, 4, 1, 1, 1)

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": la, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": la, "concat_mask": mask})
        
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)
    
class GetImagesFromBatchRanged_F2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                "start_percent": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.01}),
                }}

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "range"

    CATEGORY = "Flow2/Wan 2.1"

    def range(self, images, start_percent, end_percent):
        count = len(images)
        
        start_index = round(count * start_percent)
        end_index = round(count * end_percent)

        images = images[start_index:end_index]
        return (images, )
    
class WanSkipEndFrameImages_F2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "skip_end_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                }}

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "skip"

    CATEGORY = "Flow2/Wan 2.1"

    def skip(self, images, skip_end_frames):
        images = (images - images.min()) / (images.max() - images.min())

        if skip_end_frames > 0:
            images = images[0:-skip_end_frames]

        images = torch.clamp(images, 0.0, 1.0)
        return (images, )