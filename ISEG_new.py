    
import sys
import warnings
from functools import reduce
from operator import add
import colorsys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

#from ..base.iSeg import iSeg
#from ..base.utils import generate_distinct_colors
from featurecluster import DFC_KL
from util_iseg.cam import cam_to_label
from util_iseg.miou import format_tabs, ShowSegmentResult

warnings.filterwarnings("ignore")

import warnings

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from stable_diffusion import StableDiffusion

warnings.filterwarnings("ignore")

def generate_distinct_colors(n):
    colors = []
    if n == 1:
        return [(255, 255, 255)]
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        scaled_rgb = tuple(int(x * 255) for x in rgb)
        colors.append(scaled_rgb)
    return colors

class iSeg_base(pl.LightningModule):
    def __init__(self, config, unet, noise_scheduler, t, half=False):
        super().__init__()
        self.color = None
        self.counter = 0
        self.val_counter = 0
        self.config = config
        self.half = half
        self.save_hyperparameters(config.__dict__)

        self.stable_diffusion = StableDiffusion(
            sd_version="2.1",
            half=half,
            attention_layers_to_use=config.attention_layers_to_use,
            unet=unet,
            noise_scheduler=noise_scheduler,
            t=t, 
        )
        if self.config.rand_seed is not None:
            self.stable_diffusion.rand_seed = self.config.rand_seed

        self.checkpoint_dir = None
        self.num_parts = self.config.num_class
        torch.cuda.empty_cache()

        # class global var
        self.cls_label = []
        self.token_sel_ids = []
        self.test_t_embedding = None

        self.showsegmentresult = ShowSegmentResult(num_classes=self.num_parts + 1)

    def get_masks(self, latents, output_size):
        final_attention_map = torch.zeros(self.num_parts, output_size, output_size).to(self.device)
        (
            cross_attention_maps,
            self_attention_maps,
            difference
        ) = self.stable_diffusion.train_step(
            self.test_t_embedding,
            #image,
            t=torch.tensor(self.config.test_t),
            generate_new_noise=True,
            latents=latents,
        )
        att_map, split = self.get_att_map(cross_attention_maps, self_attention_maps)
        #final_attention_map[self.cls_label] += att_map
        return split, att_map, difference

    def process_cross_att(self, cross_attention_maps):
        weight_layer = {8: 0.0, 16: 0.7, 32: 0.3, 64: 0}
        cross_attention = []
        for key, values in cross_attention_maps.items():
            if len(values) == 0: continue
            values = values.mean(1)
            normed_attn = values / values.sum(dim=(-2, -1), keepdim=True)
            if key != 64:
                normed_attn = F.interpolate(normed_attn, size=(64, 64), mode='bilinear', align_corners=False)
            cross_attention.append(weight_layer[key] * normed_attn)
        cross_attention = torch.stack(cross_attention, dim=0).sum(0)[0]
        cross_attention = cross_attention.flatten(-2, -1).permute(1, 0)
        cross_attention = torch.stack([cross_attention[:, sel].mean(1) for sel in self.token_sel_ids], dim=1)
        return cross_attention[None]

    def get_att_map(self,
                    cross_attention_maps,
                    self_attention_maps,
                    ):
        cross_att = self.process_cross_att(cross_attention_maps).float()
        self_att = self_attention_maps[64].reshape(-1, 64 * 64, 64 * 64).permute(0, 2, 1).float()
        cross_att = torch.bmm(self_att, cross_att)
        att_map = cross_att.unflatten(dim=-2, sizes=(64, 64)).permute(0, 3, 1, 2)
        att_map = F.interpolate(att_map, size=self.config.test_mask_size, mode='bilinear', align_corners=False)
        att_map = att_map[0]
        att_map = att_map-att_map.amin(dim=(-2, -1), keepdim=True)
        att_map = att_map/att_map.amax(dim=(-2, -1), keepdim=True)

        return att_map, None

    def show_cam_on_image(self, mask, save_path):
        mask = np.uint8(255 * mask.cpu())
        mask = cv2.resize(mask, dsize=(self.config.patch_size, self.config.patch_size))
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cv2.imwrite(save_path, cam)

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        text_input = self.stable_diffusion.tokenizer(
            text, padding="max_length", max_length=self.stable_diffusion.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")
        with torch.set_grad_enabled(False):
            embedding = self.stable_diffusion.text_encoder(text_input.input_ids.cuda(),
                                                           output_hidden_states=True)[0]
            embedding = embedding.half() if self.half else embedding
        return embedding

    def test_step(self, batch, batch_idx):
        print("step of test")
        return torch.tensor(0.0)

    def on_test_end(self) -> None:
        print("end of test.")

    @staticmethod
    def get_boundry_and_eroded_mask(mask):
        kernel = np.ones((7, 7), np.uint8)
        eroded_mask = np.zeros_like(mask)
        boundry_mask = np.zeros_like(mask)
        for part_mask_idx in np.unique(mask)[1:]:
            part_mask = np.where(mask == part_mask_idx, 1, 0)
            part_mask_erosion = cv2.erode(part_mask.astype(np.uint8), kernel, iterations=1)
            part_boundry_mask = part_mask - part_mask_erosion
            eroded_mask = np.where(part_mask_erosion > 0, part_mask_idx, eroded_mask)
            boundry_mask = np.where(part_boundry_mask > 0, part_mask_idx, boundry_mask)
        return eroded_mask, boundry_mask

    @staticmethod
    def get_colored_segmentation(mask, boundry_mask, image, colors):
        boundry_mask_rgb = 0
        if boundry_mask is not None:
            boundry_mask_rgb = torch.repeat_interleave(boundry_mask[None, ...], 3, 0).type(
                torch.float
            )
            for j in range(3):
                for i in range(1, len(colors) + 1):
                    boundry_mask_rgb[j] = torch.where(
                        boundry_mask_rgb[j] == i, colors[i - 1][j] / 255, boundry_mask_rgb[j])
        mask_rgb = torch.repeat_interleave(mask[None, ...], 3, 0).type(torch.float)
        for j in range(3):
            for i in range(1, len(colors) + 1):
                mask_rgb[j] = torch.where(mask_rgb[j] == i, colors[i - 1][j] / 255, mask_rgb[j])
        if boundry_mask is not None:
            final = torch.where(boundry_mask_rgb + mask_rgb == 0, image,
                                boundry_mask_rgb * 0.7 + mask_rgb * 0.5 + image * 0.3)
            return final.permute(1, 2, 0)
        else:
            final = torch.where(mask_rgb == 0, image, mask_rgb * 0.6 + image * 0.4)
            return final.permute(1, 2, 0)


def infer_mask_helper(
    latents,
    prompt: str,
    iseg_config: dict,
    unet,
    noise_scheduler,
    t,
    output_path: str = "mask.jpg",
):
    """
    Loads iSeg, calls infer_refined_mask, then saves a visualization.
    """
    cfg = iseg_config
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = iSeg(config=cfg, unet=unet, noise_scheduler=noise_scheduler, t=t)
    if t:
        model.to(cfg.device).train()
        model.stable_diffusion.setup(cfg.device)
    else:
        model.to(cfg.device).eval()
        model.stable_diffusion.setup(cfg.device)
    if getattr(cfg, "model_file", None):
        sd = torch.load(cfg.model_file, map_location=cfg.device)
        model.load_state_dict(sd, strict=False)

    # Convert image to tensor before passing to infer_refined_mask
    #image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
    #image = image.to(cfg.device)
    #if model.half:
        #image = image.half()

    mask, difference = model.infer_refined_mask(
        latent=latents,
        prompt=prompt,
    )

    for i in range(mask.shape[0]):
        model.show_cam_on_image(mask[i].to(cfg.device), f"mask_{i}.jpg")

    def threshold_mask_by_quantile(mask: torch.Tensor, quantile: float = 0.85) -> torch.Tensor:

        flat_mask = mask.flatten(start_dim=-2)  # [..., H*W]
        thresholds = flat_mask.quantile(quantile, dim=-1, keepdim=True).unsqueeze(-1)  # [..., 1, 1]
        
        binary_mask = (mask >= thresholds).float()
        return binary_mask
    
    binary_mask = threshold_mask_by_quantile(mask, quantile=0.95)
    #print(f"Saved refined mask to {output_path}")

    for i in range(binary_mask.shape[0]):
        model.show_cam_on_image(binary_mask[i].to(cfg.device), f"bi_mask_{i}.jpg")
        
    return mask, binary_mask

class iSeg(iSeg_base):
    def __init__(self, config, unet, noise_scheduler, t, half=False):
        super().__init__(config, unet, noise_scheduler, t, half)
        self.token_start_ids = None
        self.bg_context = None
        self.index = {32: [22, 5, 21, 13, 8, 27, 28, 6, 25],
                      16: [76, 1, 43, 17, 81, 6, 44, 27, 2, 8, 22, 20, 60, 78, 12, 83, 94, 47,
                           88, 96, 3, 33, 46, 52, 77, 93, 51, 58, 0, 13, 14, 19, 34, 41, 59, 87,
                           16, 24, 28, 30, 32]} if not config.att_mean else None

        self.class_name = []
        self.all_tokens = {}

    def process_cross_att(self, cross_attention_maps):

        weight_layer = {8: 0.0, 16: 0.7, 32: 0.3, 64: 0}
        cross_attention = []
        for key, values in cross_attention_maps.items():
            if len(values) == 0 or key in [8, 64]: continue
            if self.index is None:
                values = values.mean(1)
            else:
                values = values[:, self.index[key]].mean(1)
            normed_attn = values / values.sum(dim=(-2, -1), keepdim=True)
            if key != 64:
                normed_attn = F.interpolate(normed_attn, size=(64, 64), mode='bilinear', align_corners=False)
            cross_attention.append(weight_layer[key] * normed_attn)
        cross_attention = torch.stack(cross_attention, dim=0).sum(0)[0]
        if self.config.no_use_cluster:
            dfc = DFC_KL(32, 20, 64)
            clusters, n = dfc(cross_attention)
            one_hot = F.one_hot(clusters, n)
            self_att = one_hot[:, clusters]
            cross_attention = torch.matmul(self_att.type(cross_attention.dtype),
                                           cross_attention.flatten(-2, -1).permute(1, 0))
            cross_attention = cross_attention/self_att.sum(-1, keepdim=True)
        else:
            cross_attention = cross_attention.flatten(-2, -1).permute(1, 0)
        #print(cross_attention)
        #print(cross_attention.shape())
        #cross_attention = cross_attention.unsqueeze(0)  # [1, 4096, 77]
        #cross_attention = torch.stack([cross_attention[:, sel].mean(1) for sel in self.token_sel_ids], dim=1)
        cross_attention = cross_attention[:,self.token_sel_ids]
        return cross_attention[None]

    def get_att_map(self,
                    cross_attention_maps,
                    self_attention_maps,
                    ):
        if not self.config.no_use_self_ers:
            return super().get_att_map(cross_attention_maps, self_attention_maps)
        else:
            # cross attention 特征归一化 & 上采样融合
            cross_att = self.process_cross_att(cross_attention_maps).float()
            cross_attn = cross_att - cross_att.amin(dim=-2, keepdim=True)  # cross_att: 4096, 20
            cross_attn = cross_attn / cross_attn.sum(dim=-2, keepdim=True)  # 归一化

            trans_mat = self_attention_maps[64][:, [1, 2]].mean(1).flatten(-2, -1).permute(0, 2, 1).float()
            trans_mat = trans_mat/torch.amax(trans_mat, dim=-2, keepdim=True)

            trans_mat += torch.where(trans_mat == 0, 0, self.config.ent * (torch.log10(torch.e * trans_mat)))
            trans_mat = torch.clamp(trans_mat, min=0)

            trans_mat_p = trans_mat.clone() 
            trans_mat_p = trans_mat_p/trans_mat_p.sum(dim=-1, keepdim=True)

            for _ in range(self.config.iter):
                cross_attn = torch.bmm(trans_mat_p, cross_attn)
                # cross_attn = torch.where(cross_attn < cross_attn.amax(dim=-2, keepdim=True) * 0.1, 0, cross_attn)

                cross_attn = cross_attn-cross_attn.amin(dim=-2, keepdim=True)
                cross_attn = cross_attn/cross_attn.sum(dim=-2, keepdim=True)

            cross_att = cross_attn
        att_map = cross_att.unflatten(dim=-2, sizes=(64, 64)).permute(0, 3, 1, 2)
        att_map = F.interpolate(att_map, size=self.config.test_mask_size, mode='bilinear', align_corners=False)
        att_map = att_map[0]
        att_map = att_map-att_map.amin(dim=(-2, -1), keepdim=True)
        att_map = att_map/att_map.amax(dim=(-2, -1), keepdim=True)
        #print(f'att_map: {att_map.max()}')
        return att_map, None

    def show_cam_on_image(self, mask, save_path):
        # Ensure mask is 2D and dtype uint8 in [0, 255]
        mask = mask.clone().detach().cpu()
        '''
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # from (1, H, W) → (H, W)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # from (H, W, 1) → (H, W)
        '''
        mask = mask - mask.min()
        mask = mask / (mask.max() + 1e-8)
        mask = (mask * 255).to(torch.uint8).numpy()

        # Now mask is (H, W) uint8
        mask = cv2.resize(mask, dsize=(self.config.patch_size, self.config.patch_size))

        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap / np.max(heatmap)
        cam = np.uint8(255 * cam)
        cv2.imwrite(save_path, cam)

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device)
        print(f"\nuse_self_ers: {self.config.no_use_self_ers}\nuse_cross_enh: {self.config.no_use_cross_enh}")
        self.prepare_data_name()
        self.color = generate_distinct_colors(self.config.num_class)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        text_input = self.stable_diffusion.tokenizer(
            text, padding="max_length", max_length=self.stable_diffusion.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")
        with torch.set_grad_enabled(False):
            embedding = self.stable_diffusion.text_encoder(text_input.input_ids.cuda(),
                                                           output_hidden_states=True)[0]
            embedding = embedding.half() if self.half else embedding
        return embedding

    def infer_refined_mask(self, latent, prompt):
        
    
        self.test_t_embedding = prompt#self.get_text_embedding(prompt).to(latent.device)

        self.token_sel_ids = [4,6,8,10,12]
        meaning_index = reduce(add, self.token_sel_ids)
        self.test_t_embedding[:, meaning_index] = self.test_t_embedding[:, meaning_index]*self.config.enhanced if self.config.no_use_cross_enh else 1
        print(self.test_t_embedding[:, meaning_index])
        split_mask, final_attention_map, difference = self.get_masks(latent, self.config.test_mask_size)
        #print(f'final attn map size: {final_attention_map.size()}')
        final_attention_map = F.interpolate(final_attention_map[None], size=(64,64),#(image.shape[-2:]),
                                            mode="bilinear", align_corners=False)[0]

        self.stable_diffusion.feature_maps = {}
        self.stable_diffusion.toq_maps = {}
        self.stable_diffusion.attention_maps = {}

        return final_attention_map, difference

    def on_test_end(self) -> None:
        iou = self.showsegmentresult.calculate()
        if self.config.save_file is not None:
            with open(self.config.save_file, 'a') as f:
                dat = (f"\niter:{self.config.iter} enhanced:{self.config.enhanced} "
                       f"ent:{self.config.ent}---> mIou:{iou['mIoU']}\t")
                for k, v in iou["IoU"].items():
                    dat += f" {k}:{v}"
                f.write(dat)
        cat_list = ["background"] + self.class_name
        format_tabs([iou], ['CAM'], cat_list)

