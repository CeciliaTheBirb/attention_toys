"""
Modified from:
    Break-A-Scene: https://github.com/google/break-a-scene
"""

import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
from torch.utils.data import Dataset
import numpy as np
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
#from svdiff.diffusers_models.unet_2d_condition import UNet2DConditionModel
from models.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModel_mask
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import ptp_utils
from ptp_utils import AttentionStore, compute_score, emd_distance_2d, get_connect, wasser_loss
from diffusers.models.attention import Attention as CrossAttention
import torchvision.transforms as T
from clustering.finch import FINCH
from scipy.optimize import linear_sum_assignment as linear_assignment 
from infer import infer_with_embed
from concept_utils.loss import SupConLoss
from concept_utils.pca import pca_visual
import json
from util.config import get_iseg_config
from ISEG_new import infer_mask_helper

check_min_version("0.12.0")

logger = get_logger(__name__)

def save_progress(text_encoder, placeholder_token, placeholder_token_id, accelerator, save_path):
    logger.info("Saving embeddings")
    learned_embeds_dict = {}
    for i, ph_id in enumerate(placeholder_token_id):
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[ph_id]
        learned_embeds_dict[placeholder_token[i]] = learned_embeds.detach().cpu()
    torch.save(learned_embeds_dict, save_path)
    
    return learned_embeds_dict

def test_generation(args, placeholder_token, save_path, global_step, split_state=False):
    for i, tok in enumerate(placeholder_token):
        prompt = "A photo of {}".format(tok)

        grid = infer_with_embed(save_path, args.pretrained_model_name_or_path, prompt, num_samples=args.num_samples, num_rows=args.num_rows)
        
        if not os.path.exists(os.path.join(args.output_dir, 'images')):
            os.mkdir(os.path.join(args.output_dir, 'images'))
            
        grid.save(os.path.join(args.output_dir, 'images/' + prompt.replace(' ', '-') + '-step-{}.png'.format(global_step)))
    
    if not split_state:
        full_prompt = "A photo of " + " and ".join(placeholder_token)
        grid = infer_with_embed(save_path, args.pretrained_model_name_or_path, full_prompt, num_samples=args.num_samples, num_rows=args.num_rows)
            
        grid.save(os.path.join(args.output_dir, 'images/' + full_prompt.replace(' ', '-') + '-step-{}.png'.format(global_step)))

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo at the beach",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--no_prior_preservation",
        action="store_false",
        help="Flag to add prior preservation loss.",
        dest="with_prior_preservation"
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_train_text_encoder",
        action="store_false",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
        dest="train_text_encoder"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--phase1_train_steps",
        type=int,
        default="500",
        help="Number of trainig steps for the first phase.",
    )
    parser.add_argument(
        "--phase2_train_steps",
        type=int,
        default="0",
        help="Number of trainig steps for the second phase.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=5e-4,
        help="The LR for the Textual Inversion steps.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument("--lambda_attention", type=float, default=1e-2)
    parser.add_argument("--img_log_steps", type=int, default=200)
    parser.add_argument("--num_of_assets", type=int, default=1)
    parser.add_argument("--initializer_tokens", type=str, nargs="+", default=[])
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<asset>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--do_not_apply_masked_loss",
        action="store_false",
        help="Use masked loss instead of standard epsilon prediciton loss",
        dest="apply_masked_loss"
    )
    parser.add_argument(
        "--log_checkpoints",
        action="store_true",
        help="Indicator to log intermediate model checkpoints",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=1,
        help="Number of rows to generate.",
    )
    parser.add_argument(
        "--init_merge_rand",
        action="store_true",
        help="Whether merge random tokens for initialization.",
    )
    parser.add_argument(
        "--num_split_tokens",
        type=int,
        default=5,
        help="The number we split the tokens.",
    )
    parser.add_argument(
        "--weight_contrast",
        type=float,
        default=1.0,
        help="The weight of contrastive loss.",
    )
    parser.add_argument(
        "--merge_step",
        type=int,
        default=100,
        help="The step we merge tokens"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="The temperature of supervised contrastive loss."
    )
    parser.add_argument(
        "--vis_pca",
        action="store_true",
        default=False,
        help="whether to visualize pca or not",
    )
        
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.initializer_tokens = []
   
    assert len(args.initializer_tokens) == 0 or len(args.initializer_tokens) == args.num_of_assets
    args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )
        
    return args

class TokenManager():
    def __init__(self, placeholder_tokens, tokenizer, num_split_tokens):
        self.all_ph_tokens = placeholder_tokens
        self.num_tokens = len(self.all_ph_tokens)
        
        self.mask_list = None
        self.feat_list = None
            
        self.ph_tokens_used = self.all_ph_tokens
        
        self.tokenizer = tokenizer
        self.num_split_tokens = num_split_tokens
        self.split_state = False
        
    def update_mask(self, mask_list_new, feat_list_new, flip):
        if flip[0]:
            mask_list_new = self.flip_mask(mask_list_new) 
            feat_list_new = self.flip_mask(feat_list_new)
        
        if self.mask_list is None:
            self.mask_list, self.feat_list = mask_list_new, feat_list_new
            self.ph_tokens_used = self.all_ph_tokens[:len(self.mask_list)]
        else:
            self.old_to_new(mask_list_new, feat_list_new)
            
        self.num_tokens = len(self.mask_list)
    
    def old_to_new(self, mask_list_new, feat_list_new):
        
        num_old = len(self.mask_list)
        num_new = len(mask_list_new)
        
        feat_mat_old = torch.stack(self.feat_list, dim=0)
        feat_mat_new = torch.stack(feat_list_new, dim=0)
        
        ## avg pooling
        feat32_new = feat_mat_new.reshape(feat_mat_new.shape[0], -1, 64, 64)
        feat32_old = feat_mat_old.reshape(feat_mat_old.shape[0], -1, 64, 64)
        
        feat32_new = F.avg_pool2d(feat32_new, kernel_size=2, stride=2).reshape(-1, 32*32)
        feat32_old = F.avg_pool2d(feat32_old, kernel_size=2, stride=2).reshape(-1, 32*32)

        feat32_new = feat32_new.cpu().numpy()
        feat32_old = feat32_old.cpu().numpy()
        
        emd_distance = emd_distance_2d(np.float32(feat32_old), np.float32(feat32_new))
        
        row_ind, col_ind = linear_assignment(emd_distance)
        
        self.mask_list[row_ind] = mask_list_new[col_ind]
        self.feat_list[row_ind] = feat_list_new[col_ind]
        
        if num_old < num_new:
            col_ind_not = [i for i in range(num_new) if i not in col_ind]
            self.mask_list = self.mask_list + mask_list_new[col_ind_not]
            self.feat_list = self.feat_list + feat_list_new[col_ind_not]
            
            self.ph_tokens_used = self.all_ph_tokens[:num_new]
    
    def flip_mask(self, input_list):
        shape = input_list[0].shape
        output_list = [TF.hflip(mi.reshape(1, 512, 512)).reshape(shape) for mi in input_list]
        
        return output_list
    
    def current_tokens(self, tokenizer):
        ph_tokens = self.ph_tokens_used
        ph_tokens_ids = [tokenizer.convert_tokens_to_ids(ph_tokens[i]) for i in range(len(ph_tokens))]
        return ph_tokens, ph_tokens_ids
    
    def return_single_token(self, tokens_ids_to_use, flip, bsz):
        tokens_to_use = [self.ph_tokens_used[tkn_i] for tkn_i in tokens_ids_to_use]
        prompt = "a photo of " + " and ".join(tokens_to_use)
        masks_to_use = [self.mask_list[tkn_i] for tkn_i in tokens_ids_to_use]
        feats_to_use = [self.feat_list[tkn_i] for tkn_i in tokens_ids_to_use]
        
        token_ids = torch.tensor(tokens_ids_to_use)
        
        if flip[0]:
            masks_to_use = self.flip_mask(masks_to_use)
            feats_to_use = self.flip_mask(feats_to_use)
        
        prompt_ids = self.tokenizer(
            [prompt] * bsz,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        masks_to_use = torch.stack(masks_to_use, dim=0)
        feats_to_use = torch.stack(feats_to_use, dim=0)
        
        return prompt_ids, tokens_to_use, masks_to_use, feats_to_use, token_ids
    
    def split_tokens(self):
        if not self.split_state:
            self.ph_tokens_used = self.all_ph_tokens[:self.num_tokens * self.num_split_tokens]
            self.mask_list = self.mask_list  * self.num_split_tokens
            self.feat_list = self.feat_list * self.num_split_tokens
            self.split_state = True
            
    def merge_tokens(self):
        if self.split_state:
            self.ph_tokens_used = self.all_ph_tokens[:self.num_tokens]
            self.mask_list = self.mask_list[:self.num_tokens]
            self.feat_list = self.feat_list[:self.num_tokens]
            self.split_state = False
    
    def get_token_num(self):
        return self.num_tokens
    
    def loader(self, flip, bsz):
        prompt_ids_list = []
        tokens_to_use_list = []
        masks_to_use_list = []
        feats_to_use_list = []
        token_ids_list = []
        
        for i in range(len(self.ph_tokens_used)):
            prompt_ids, tokens_to_use, masks_to_use, feats_to_use, token_ids = self.return_single_token([i], flip, bsz)
            prompt_ids_list.append(prompt_ids)
            tokens_to_use_list.append(tokens_to_use)
            masks_to_use_list.append(masks_to_use)
            feats_to_use_list.append(feats_to_use)
            token_ids_list.append(token_ids)
        
        if not self.split_state:
            prompt_ids, tokens_to_use, masks_to_use, feats_to_use, token_ids = self.return_single_token(list(range(len(self.ph_tokens_used))), flip, bsz)
            prompt_ids_list.append(prompt_ids)
            tokens_to_use_list.append(tokens_to_use)
            masks_to_use_list.append(masks_to_use)
            feats_to_use_list.append(feats_to_use)
            token_ids_list.append(token_ids)
        
        return prompt_ids_list, tokens_to_use_list, masks_to_use_list, feats_to_use_list, token_ids_list
        
class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
        self,
        instance_data_root,
        tokenizer,
        prompt,
        size=512,
        center_crop=False,
        flip_p=0.5,
    ):
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.prompt=prompt
        self.tokenizer = tokenizer
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize([size,size]),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        '''
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        '''
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        instance_img_path = os.path.join(instance_data_root, "img.jpg")
        self.instance_image = self.image_transforms(Image.open(instance_img_path))
        self.instance_image_pil = Image.open(instance_img_path)

        self._length = 1
        self.null_prompt = ""

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        example["instance_images"] = self.instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        '''        
        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["flip"] = True
        else:
           ''' 
        example["flip"] = False

        return example

def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    input_ids = [example["instance_prompt_ids"] for example in examples]
    flip = [example["flip"] for example in examples]

    #pixel_values = pixel_values + pixel_values

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat(input_ids, dim=0)
    batch = {
        "pixel_values": pixel_values,
        "flip": flip,
        "input_ids": input_ids,
    }
    return batch

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


class ConceptExpress:
    def __init__(self):
        self.args = parse_args()
        self.main()

    def main(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        project_config = ProjectConfiguration(logging_dir=logging_dir)

        # Pass the project_config into Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=project_config
        )

        if (
            self.args.train_text_encoder
            and self.args.gradient_accumulation_steps > 1
            and self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)
            # seed_torch(self.args.seed)

        # Handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            with open(os.path.join(self.args.output_dir, 'args.json'), 'w') as f:
                json.dump(self.args.__dict__, f, indent=2)

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.args.pretrained_model_name_or_path, self.args.revision
        )

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )
        self.text_encoder_for_mask = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )
        self.unet_for_mask = UNet2DConditionModel_mask.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )
        # Load the tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, revision=self.args.revision, use_fast=False
            )
        elif self.args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.args.revision,
                use_fast=False,
            )
                # 1) Predefined subject tokens
        self.placeholder_tokens = ["<parrot>", "<chicken>"]#['<suit>','<shoes>','<pants>']#["<sheep>", "<cow>", "<pig>", "<chicken>", "<dog>"]
        self.args.num_of_assets = len(self.placeholder_tokens)

        # 2) Add to tokenizer
        num_added_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.placeholder_tokens}
        )
        assert num_added_tokens == self.args.num_of_assets

        # 3) Resize embeddings
        self.text_encoder_for_mask.resize_token_embeddings(len(self.tokenizer))
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        # 4) Get token IDs
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.placeholder_tokens
        )

        # 5) Initialize each with its semantic word vector
        token_embeds = self.text_encoder_for_mask.get_input_embeddings().weight.data
        for idx, tok in enumerate(self.placeholder_tokens):
            word = tok.strip("<>")
            word_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
            token_embeds[self.placeholder_token_ids[idx]] = token_embeds[word_id].detach().clone()

        words=["parrot", "chicken"]#['suit','shoes','pants']#["sheep", "cow", "pig", "chicken", "dog"]#["parrot","chicken"]
        # 6) Save the prompt used to condition the model
        self.args.instance_prompt = "a photo of " + " and ".join(words)
            
        # Set validation scheduler for logging
        self.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.validation_scheduler.set_timesteps(50)

        # We start by only optimizing the embeddings
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.unet_for_mask.requires_grad_(False)
        self.text_encoder_for_mask.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # We start by only optimizing the embeddings
        params_to_optimize = self.text_encoder.get_input_embeddings().parameters()
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.initial_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            tokenizer=self.tokenizer,
            prompt=self.args.instance_prompt,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples),
            num_workers=self.args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
            * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps
            * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        (
            self.unet,
            self.unet_for_mask,
            self.text_encoder,
            self.text_encoder_for_mask,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.unet_for_mask, self.text_encoder, self.text_encoder_for_mask, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if (
            self.args.train_text_encoder
            and self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        if len(self.args.initializer_tokens) > 0:
            # Only for logging
            self.args.initializer_tokens = ", ".join(self.args.initializer_tokens)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        
        def clean_config(args_dict):
            cleaned = {}
            for k, v in args_dict.items():
                if isinstance(v, (int, float, str, bool)):
                    cleaned[k] = v
                elif isinstance(v, Path):
                    cleaned[k] = str(v)
                # skip or stringify anything else
                else:
                    cleaned[k] = str(v)
            return cleaned

        config = clean_config(vars(self.args))
        self.accelerator.init_trackers("dreambooth", config=config)
        #if self.accelerator.is_main_process:
            #self.accelerator.init_trackers("dreambooth", config=vars(self.args))
        '''
        prompt = "a photo of a suit in the snow"
        from try_ import generate_image
        from torchvision.utils import save_image
        image = generate_image(self.unet, self.vae, self.tokenizer, self.text_encoder, lr_scheduler, prompt)
        save_image(image, "suit_in_snow.png")
        '''
        # Train
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * self.args.gradient_accumulation_steps
                )

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        # keep original embeddings as reference
        orig_embeds_params = (
            self.accelerator.unwrap_model(self.text_encoder)
            .get_input_embeddings()
            .weight.data.clone()
        )

        # Create attention controller
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)
        
        self.token_manager = TokenManager(self.placeholder_tokens, self.tokenizer, self.args.num_split_tokens)
        
        self.contrastive_loss = SupConLoss(temperature=self.args.temperature, 
                                           base_temperature=self.args.temperature)
        
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                if self.args.phase1_train_steps == global_step:
                    self.unet.requires_grad_(True)
                    if self.args.train_text_encoder:
                        self.text_encoder.requires_grad_(True)
                    unet_params = self.unet.parameters()

                    params_to_optimize = (
                        itertools.chain(unet_params, self.text_encoder.parameters())
                        if self.args.train_text_encoder
                        else itertools.chain(
                            unet_params,
                            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().parameters(),
                        )
                    )
                    del optimizer
                    optimizer = optimizer_class(
                        params_to_optimize,
                        lr=self.args.learning_rate,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        weight_decay=self.args.adam_weight_decay,
                        eps=self.args.adam_epsilon,
                    )
                    del lr_scheduler
                    lr_scheduler = get_scheduler(
                        self.args.lr_scheduler,
                        optimizer=optimizer,
                        num_warmup_steps=self.args.lr_warmup_steps
                        * self.args.gradient_accumulation_steps,
                        num_training_steps=self.args.max_train_steps
                        * self.args.gradient_accumulation_steps,
                        num_cycles=self.args.lr_num_cycles,
                        power=self.args.lr_power,
                    )
                    optimizer, lr_scheduler = self.accelerator.prepare(
                        optimizer, lr_scheduler
                    )
                
                if global_step == 0:   
                    with torch.no_grad():
                        latents = self.vae.encode(
                            batch["pixel_values"].to(dtype=self.weight_dtype)
                        ).latent_dist.sample()
                        latents = latents * 0.18215
                        bsz = latents.shape[0]
                        iseg_config = get_iseg_config()
                        # 1. Tokenize the input prompt
                        inputs = self.tokenizer(
                            self.args.instance_prompt,
                            padding='max_length',
                            truncation=True,
                            max_length=self.tokenizer.model_max_length,
                            return_tensors="pt"
                        ).input_ids.to(self.accelerator.device)
                        #print(inputs)

                        # 2. Feed into the text encoder
                        encoder_hidden_states = self.text_encoder_for_mask(batch["input_ids"])[0]

                        feat_list, mask_list = infer_mask_helper(latents.detach().clone(), encoder_hidden_states.detach().clone(), iseg_config, self.unet_for_mask, self.noise_scheduler, False)
                        self.token_manager.update_mask(mask_list, feat_list, batch["flip"])

                if global_step == 0:
                    subject_tokens = self.placeholder_tokens
                    self.token_manager.ph_tokens_used = subject_tokens
                '''                    
                if global_step == self.args.merge_step: 
                    save_path = os.path.join(self.args.output_dir, "learned_embeds_init.bin")
                    learned_embeds_dict = save_progress(
                        self.text_encoder,
                        subject_tokens,
                        token_ids,
                        self.accelerator,
                        save_path
                    )
                    if self.accelerator.is_main_process:
                        # stage="init" just labels your outputs
                        # split_state=None because you’re not splitting/merging
                        test_generation(
                            self.args,
                            subject_tokens,
                            learned_embeds_dict,
                            global_step=global_step,
                            #stage="init",
                            split_state=None)
                    '''
                self.accelerator.wait_for_everyone()
                prompt_ids_list, tokens_to_use_list, masks_to_use_list, feats_to_use_list, token_ids_list = self.token_manager.loader(batch["flip"], bsz)
                
                logs = {}
                
                ############# pca ###############
                if self.args.vis_pca and self.token_manager.split_state:
                    converted_ids = self.tokenizer.encode([i[0] for i in tokens_to_use_list], 
                                                            add_special_tokens=False, return_tensors='pt')
                    
                    sample_embeddings = self.accelerator.unwrap_model(
                                self.text_encoder
                            ).get_input_embeddings()(converted_ids.to(self.text_encoder.device))[0]
                    
                    label = torch.tensor(
                        list(range(self.token_manager.get_token_num())) * self.args.num_split_tokens, dtype=torch.int
                        ).to(sample_embeddings.device)
                    
                    sample_embeddings_normalized = F.normalize(sample_embeddings, p=2, dim=-1)
                    
                    if not os.path.exists(os.path.join(self.args.output_dir, 'pca_vis')):
                        os.makedirs(os.path.join(self.args.output_dir, 'pca_vis'), exist_ok=True)
                    
                    pca_visual(sample_embeddings_normalized, label, global_step, self.args.output_dir)
                #################
                
                # Skip steps until we reach the resumed step
                if (
                    self.args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):
                    for list_idx in range(len(prompt_ids_list)):
                        prompt_ids, tokens_to_use, masks_to_use, feats_to_use, token_ids = \
                            prompt_ids_list[list_idx], tokens_to_use_list[list_idx],\
                                masks_to_use_list[list_idx], feats_to_use_list[list_idx], token_ids_list[list_idx]
                        # Convert images to latent space
                        latents = self.vae.encode(
                            batch["pixel_values"].to(dtype=self.weight_dtype)
                        ).latent_dist.sample()
                        latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        )
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = self.noise_scheduler.add_noise(
                            latents, noise, timesteps
                        )

                        # Get the text embedding for conditioning
                        encoder_hidden_states = self.text_encoder(prompt_ids.to(latents.device))[0]
                        # Predict the noise residual
                        model_pred = self.unet(
                            noisy_latents, timesteps, encoder_hidden_states
                        ).sample

                        # Get the target for loss depending on the prediction type
                        if self.noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif self.noise_scheduler.config.prediction_type == "v_prediction":
                            target = self.noise_scheduler.get_velocity(
                                latents, noise, timesteps
                            )
                        else:
                            raise ValueError(
                                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                            )
                        
                        #print("model_pred.shape:", model_pred.shape)
                        #_, model_pred = torch.chunk(model_pred, 2, dim=0)
                        #_, target = torch.chunk(target, 2, dim=0)
                            
                        if self.args.apply_masked_loss:
                            max_mask = torch.max(
                                masks_to_use, dim=0, keepdim=True
                            ).values.unsqueeze(1)
                            max_mask_np = T.ToPILImage()(max_mask.reshape(64,64))
                            pil = (batch["pixel_values"][0] * 0.5 + 0.5) 
                            pil = T.ToPILImage()(pil)
                            image_masked_save = self.vis_masked_image(pil, max_mask_np)

                            model_pred = model_pred * max_mask
                            target = target * max_mask
                            
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                        # Attention loss
                        attn_loss = 0.
                        create=0
                        for batch_idx in range(self.args.train_batch_size):
                            feats_to_use = feats_to_use.reshape(-1,64,64)
                            GT_feats = F.interpolate(
                                input=feats_to_use.unsqueeze(1), size=(16, 16)
                            )

                            GT_masks = F.interpolate(
                                input=masks_to_use.unsqueeze(1), size=(16, 16)
                            )
                            agg_attn = self.aggregate_attention(
                                res=16,
                                from_where=("up", "down"),
                                is_cross=True,
                                select=batch_idx,
                            )
                            
                            curr_cond_batch_idx = self.args.train_batch_size + batch_idx
                            
                            for mask_id in range(len(GT_feats)):
                                curr_placeholder_token_id = self.placeholder_token_ids[
                                    token_ids[mask_id]
                                ]

                                asset_idx = (
                                    (
                                        prompt_ids[batch_idx]
                                        == curr_placeholder_token_id
                                    )
                                    .nonzero()
                                    .item()
                                )
                                asset_attn_mask = agg_attn[..., asset_idx]
                                feat_target = GT_feats[mask_id, 0].detach()
                                
                                attn_loss += wasser_loss(
                                    feat_target.float(),
                                    asset_attn_mask.float(),
                                )
                                
                                asset_attn_mask1 = asset_attn_mask.reshape(-1)
                                
                                transform = T.ToPILImage()
                                attn_norm = asset_attn_mask1 / asset_attn_mask1.max()
                                attn_norm = transform(attn_norm.reshape(16,16))
                                
                                if not os.path.exists(os.path.join(self.args.output_dir, 'update_attn')):
                                    os.makedirs(os.path.join(self.args.output_dir, 'update_attn'), exist_ok=True)
                                    
                                attn_norm.save(
                                    os.path.join(self.args.output_dir, 
                                                    'update_attn/attn_{}.png'.format(curr_placeholder_token_id)
                                                    )
                                    )

                        if self.token_manager.split_state: # split
                            attention_weight = self.args.lambda_attention
                        else: # merge
                            attention_weight = self.args.lambda_attention
                            
                        attn_loss = attention_weight * (
                            attn_loss / self.args.train_batch_size
                        )
                        
                        logs["attn_loss"] = attn_loss.detach().item()
                        loss += attn_loss
                        
                        if self.token_manager.split_state:
                            converted_ids = self.tokenizer.encode([i[0] for i in tokens_to_use_list], 
                                                                  add_special_tokens=False, return_tensors='pt')
                            
                            sample_embeddings = self.accelerator.unwrap_model(
                                        self.text_encoder
                                    ).get_input_embeddings()(converted_ids.to(self.text_encoder.device))[0]
                            
                            label = torch.tensor(
                                list(range(self.token_manager.get_token_num())) * self.args.num_split_tokens, dtype=torch.int
                                ).to(sample_embeddings.device)
                            
                            sample_embeddings_normalized = F.normalize(sample_embeddings.unsqueeze(1), p=2, dim=-1)
                            
                            loss_con = self.contrastive_loss(sample_embeddings_normalized, labels=label)
                            
                            loss += loss_con * self.args.weight_contrast

                        self.accelerator.backward(loss)

                        # No need to keep the attention store
                        self.controller.attention_store = {}
                        self.controller.cur_step = 0

                        if self.accelerator.sync_gradients:
                            params_to_clip = (
                                itertools.chain(
                                    self.unet.parameters(), self.text_encoder.parameters()
                                )
                                if self.args.train_text_encoder
                                else self.unet.parameters()
                            )
                            self.accelerator.clip_grad_norm_(
                                params_to_clip, self.args.max_grad_norm
                            )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)
                        
                        if global_step < self.args.phase1_train_steps:
                            with torch.no_grad():
                                self.accelerator.unwrap_model(
                                    self.text_encoder
                                ).get_input_embeddings().weight[
                                    : -self.args.num_of_assets
                                ] = orig_embeds_params[
                                    : -self.args.num_of_assets
                                ]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(self.args.output_dir, f"learned_embeds-step-{global_step}.bin")
                            placeholder_token, placeholder_token_id = self.token_manager.current_tokens(self.tokenizer)
                            
                            learned_embeds_dict = save_progress(self.text_encoder, placeholder_token, placeholder_token_id, self.accelerator, save_path)
                            test_generation(self.args, placeholder_token, learned_embeds_dict, global_step, self.token_manager.split_state)
                            
                    if (
                        self.args.log_checkpoints
                        and global_step % self.args.img_log_steps == 0
                        and global_step > self.args.phase1_train_steps
                    ):
                        ckpts_path = os.path.join(
                            self.args.output_dir, "checkpoints", f"{global_step:05}"
                        )
                        os.makedirs(ckpts_path, exist_ok=True)
                        self.save_pipeline(ckpts_path)

                        img_logs_path = os.path.join(self.args.output_dir, "img_logs")
                        os.makedirs(img_logs_path, exist_ok=True)

                        if self.args.lambda_attention != 0:
                            self.controller.cur_step = 1
                            last_sentence = prompt_ids[batch_idx]
                            last_sentence = last_sentence[
                                (last_sentence != 0)
                                & (last_sentence != 49406)
                                & (last_sentence != 49407)
                            ]
                            last_sentence = self.tokenizer.decode(last_sentence)
                            self.save_cross_attention_vis(
                                last_sentence,
                                attention_maps=agg_attn.detach().cpu(),
                                path=os.path.join(
                                    img_logs_path, f"{global_step:05}_step_attn.jpg"
                                ),
                            )
                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                        self.perform_full_inference(
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_full_pred.jpg"
                            )
                        )
                        full_agg_attn = self.aggregate_attention(
                            res=16, from_where=("up", "down"), is_cross=True, select=0
                        )
                        self.save_cross_attention_vis(
                            self.args.instance_prompt,
                            attention_maps=full_agg_attn.detach().cpu(),
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_full_attn.jpg"
                            ),
                        )
                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                self.accelerator.wait_for_everyone()
                
                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break
        
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.args.output_dir, f"learned_embeds_final.bin")
            placeholder_token, placeholder_token_id = self.token_manager.current_tokens(self.tokenizer)
            
            learned_embeds_dict = save_progress(self.text_encoder, placeholder_token, placeholder_token_id, self.accelerator, save_path)
            test_generation(self.args, placeholder_token, learned_embeds_dict, global_step)

        self.accelerator.end_training()

    def save_pipeline(self, path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                revision=self.args.revision,
            )
            pipeline.save_pretrained(path)

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count
        
    def vis_masked_image(self, image, mask):
        tsfm = transforms.Resize([256,256])
        image_np = np.array(tsfm(image))
        
        
        tsfm_mask = transforms.Resize([256,256], interpolation=transforms.InterpolationMode.NEAREST)
        mask_np = np.array(tsfm_mask(mask)).reshape(256,256,1) / 255
        
        image_masked = image_np * mask_np.astype(np.uint8)
        
        image_masked = Image.fromarray(image_masked)
        
        return image_masked
        
        
    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(
        self, res: int, from_where: List[str], is_cross: bool, select: int
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        self.args.train_batch_size, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def get_self_attention(self, eot_attn_mask, pil, global_step):
        pil = (pil * 0.5 + 0.5)
        out = []
        rw = 0.
        attention_maps = self.get_average_attention()
        for location in ("down", 'mid', "up"):
            for item in attention_maps[f"{location}_self"]:
                res = int(item.shape[-1] ** 0.5)
                self_maps = item.reshape(
                        self.args.train_batch_size, -1, item.shape[-2], item.shape[-1]
                )
                self_maps = self_maps.sum(1) / self_maps.shape[1]
                self_maps = self_maps.reshape(self.args.train_batch_size, -1, res, res)
                upsampled_self_maps = F.interpolate(
                    input=self_maps, size=(64, 64), mode='bilinear'
                )
                upsampled_self_maps = upsampled_self_maps.reshape(self.args.train_batch_size, res, res, -1)

                upsampled_self_maps = torch.repeat_interleave(upsampled_self_maps, repeats=int(64/res), dim=1)
                upsampled_self_maps = torch.repeat_interleave(upsampled_self_maps, repeats=int(64/res), dim=2)
                
                upsampled_self_maps = upsampled_self_maps.reshape(self.args.train_batch_size, 64**2, 64**2)
                
                r_weight = float(res/64)
                
                out.append(upsampled_self_maps * r_weight)
                rw += r_weight
        
        out = torch.stack(out, dim=0)
        out = out.sum(0) / rw
        
        feat_map = out.reshape(64**2, 64**2)
        mask_list, feat_list = self.cluster_attention(feat_map, eot_attn_mask, pil, global_step)
        
        # assert False, "EXIT!"
        return mask_list, feat_list
    
    def cluster_attention(self, feat_map, eot_attn_mask, pil, global_step): # (HW) * (HW)
        
        if not os.path.exists(os.path.join(self.args.output_dir, "attention/{}-step".format(global_step))):
            os.makedirs(os.path.join(self.args.output_dir, "attention/{}-step".format(global_step)), exist_ok=True)
            
        # x = F.softmax(feat_map / 0.05, dim=-1)
        # x = feat_map / torch.norm(feat_map, dim=-1, keepdim=True)
        x = feat_map / feat_map.sum(-1, keepdim=True)
        
        x_np = x.detach().cpu().numpy()
        c, num_clust, req_c, min_sim_init = FINCH(x_np, initial_rank=None, 
                                    req_clust=None, distance='kld', 
                                    ensure_early_exit=False, verbose=True)
        
        for i, num in enumerate(num_clust):
            if num >= 10 and num_clust[i+1] < 10:
                index = i
                break
             
        out = torch.from_numpy(c[:,index])
        min_sim_init = min_sim_init[index]
        
        print("min_sim_init: {}".format(min_sim_init))
        
        eot_attn_mask = F.interpolate(
            input=eot_attn_mask[None, None], size=(64, 64), mode='bilinear'
        )
        eot_attn_mask = eot_attn_mask.reshape(64,64)
        
        eot_attn_mask = (eot_attn_mask - eot_attn_mask.min()) / (eot_attn_mask.max() - eot_attn_mask.min())
        
        transform = T.ToPILImage()
        mask = transform(eot_attn_mask)
        mask.save(os.path.join(self.args.output_dir, 'attention/{}-step/eot_attention.png'.format(global_step)))
        
        pil = transform(pil)
        pil.save(os.path.join(self.args.output_dir, 'attention/{}-step/image.png'.format(global_step)))
        
        mask_candidate = []
        feat_candidate = []
        feat_max_candidate = []
        
        for i in range(out.max()+1):
            mask = torch.zeros(64**2).to(self.accelerator.device)
            mask[out==i] = 1
            mask = mask.reshape(64,64)

            score = compute_score(mask, eot_attn_mask)
            
            ### save all masks
            mask_pil = transform(mask)
            mask_pil.save(os.path.join(self.args.output_dir, 'attention/{}-step/mask_phase1_all{}.png'.format(global_step, i)))
            ##################
            
            if score > 1.0:
                mean = x[out==i].mean(0)
                feat_max = x[out==i].max(0)[0]
                
                mask_candidate.append(mask) # 64 * 64
                feat_candidate.append(mean) # 64**2
                feat_max_candidate.append(feat_max)
                
                mask = transform(mask)
                mask.save(os.path.join(self.args.output_dir, 'attention/{}-step/self-attention_candidate{}.png'.format(global_step, i)))
                
                image_masked = self.vis_masked_image(pil, mask)
                
                image_masked.save(os.path.join(self.args.output_dir, 'attention/{}-step/masked_candidate{}.png'.format(global_step, i)))
                
                norm = mean / mean.max()
                norm = transform(norm.reshape(64,64))
                norm.save(os.path.join(self.args.output_dir, 'attention/{}-step/self-attention-mean{}.png'.format(global_step, i)))
        
        mask_mat = torch.stack(mask_candidate, dim=0).detach()
        feat_mat = torch.stack(feat_candidate, dim=0).detach()
        feat_max_mat = torch.stack(feat_max_candidate, dim=0).detach()

        feat_mat_np = feat_mat.cpu().numpy()
        c_2, num_clust, req_c, min_sim_list_cache = FINCH(feat_mat_np, initial_rank=None, 
                                    req_clust=None, distance='kld', 
                                    ensure_early_exit=True, verbose=True, 
                                    mask_candidate=mask_mat.detach().cpu().numpy(),
                                    min_sim=min_sim_init)
        
        c_2 = torch.from_numpy(c_2[:,-1])
        
        mask_final = []
        feat_final = []
        
        for i in range(c_2.max()+1):
            mask_i = mask_mat[c_2==i].sum(dim=0)
            mask_final.append(mask_i)
            
            feat_i = feat_mat[c_2==i].mean(dim=0)
            feat_final.append(feat_i)
            
            # feat_i = feat_max_mat[c_2==i].max(0)[0]
            # feat_final.append(feat_i)
            
            mask_i = transform(mask_i)
            mask_i.save(os.path.join(self.args.output_dir,'attention/{}-step/final_attention{}.png'.format(global_step, i)))
            
            image_masked = self.vis_masked_image(pil, mask_i)
            
            image_masked.save(os.path.join(self.args.output_dir,'attention/{}-step/final_masked{}.png'.format(global_step, i)))
            
            norm = feat_i / feat_i.max()
            norm = transform(norm.reshape(64,64))
            norm.save(os.path.join(self.args.output_dir,'attention/{}-step/final_mean{}.png'.format(global_step, i)))
        
        return mask_final, feat_final
        
    @torch.no_grad()
    def perform_full_inference(self, path, guidance_scale=7.5):
        self.unet.eval()
        self.text_encoder.eval()

        latents = torch.randn((1, 4, 64, 64), device=self.accelerator.device)
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.accelerator.device)
        input_ids = self.tokenizer(
            [self.args.instance_prompt],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        cond_embeddings = self.text_encoder(input_ids)[0]
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        images = self.vae.decode(latents.to(self.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.unet.train()
        if self.args.train_text_encoder:
            self.text_encoder.train()

        Image.fromarray(images[0]).save(path)

    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, attention_maps, path):
        tokens = self.tokenizer.encode(prompt)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(
                image, self.tokenizer.decode(int(tokens[i]))
            )
            images.append(image)
        vis = ptp_utils.view_images(np.stack(images, axis=0))
        vis.save(path)

class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        batch_size = hidden_states.shape[0]
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


if __name__ == "__main__":
    ConceptExpress()
