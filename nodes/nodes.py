from comfy.utils import common_upscale
import torch
import comfy.model_management as mm
import node_helpers
import nodes

# reference: https://github.com/raindrop313/ComfyUI-WanVideoStartEndFrames
# reference: https://github.com/kijai/ComfyUI-WanVideoWrapper/

class WanVaeWrapper_F2:
    def __init__(self, wv, end_image):
        self.wv = wv
        self.end_image = end_image

    def __getattr__(self, name):
        return getattr(self.wv, name)

    def encode(self, x):
        if self.end_image is not None:
            mu = self.encode_2(x)
        else:
            mu = self.encode_1(x)
        return mu
    
    def decode(self, x):
        if self.end_image is not None:
            mu = self.decode_2(x)
        else:
            mu = self.decode_1(x)
        return mu
    
    def encode_1(self, x):
        self.clear_cache()

        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        self.clear_cache()
        return mu

    def encode_2(self, x):
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
    
    def decode_1(self, z):
        self.clear_cache()

        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out
    
    def decode_2(self, z):
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
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "start_image": ("IMAGE", ),
                },
                "optional": {"end_image": ("IMAGE", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "Flow2/Wan 2.1"

    def encode(self, positive, negative, vae, width, height, length, start_image=None, end_image=None, clip_vision_output=None):
        device = mm.intermediate_device()

        valid_end_image = end_image is not None

        resized_start_image = common_upscale(start_image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        if valid_end_image:
            resized_end_image = common_upscale(end_image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)

        empty_frames = torch.ones(length - 1, height, width, 3, device=device) * 0.5

        if valid_end_image:
            concatenated = torch.cat([resized_start_image.to(device), empty_frames, resized_end_image.to(device)], dim=0)
        else:
            concatenated = torch.cat([resized_start_image.to(device), empty_frames], dim=0)

        if valid_end_image: # Replacing comfyui method can cause problems.
            vae.first_stage_model.encode = WanVaeWrapper_F2(vae.first_stage_model, end_image).encode

        la = vae.encode(concatenated)
        print(1 + bool(valid_end_image))
        latent = torch.zeros([1, 16, ((length - 1) // 4) + (1 + bool(valid_end_image)), height // 8, width // 8], device=device)

        mask = torch.ones((1, 1, latent.shape[2], la.shape[-2], la.shape[-1]), device=device)
        mask[:, :, :1] = 0.0
        if valid_end_image:
            mask[:, :, -1:] = 0.0

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": la, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": la, "concat_mask": mask})
        
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)