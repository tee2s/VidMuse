# Modified from Audiocraft (https://github.com/facebookresearch/audiocraft)

from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn

from ..utils import utils
from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn

import time
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms._transforms_video")
import torch.nn.init as init
import os

import logging
import random
import sys
import einops
from .transformer_module import Attention, PreNorm, FeedForward
from transformers import AutoProcessor, CLIPVisionModelWithProjection, VideoMAEModel

logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]

def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """LM layer initialization.
    Inspired from xlformers: https://github.com/fairinternal/xlformers

    Args:
        method (str): Method name for init function. Valid options are:
            'gaussian', 'uniform'.
        input_dim (int): Input dimension of the initialized module.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
    """
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std  # ensure the standard deviation is std
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    """Wrapper around `get_init_fn for proper initialization of LM modules.

    Args:
        m (nn.Module): Module to initialize.
        method (str): Method name for the init function.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
        zero_bias_init (bool): Whether to initialize the bias to 0 or not.
    """
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with scale).
    """
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, x1, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.depth = x1 // num_heads

        self.query = nn.Linear(x1, x1)
        self.key = nn.Linear(x1, x1)
        self.value = nn.Linear(x1, x1)

        self.final_linear = nn.Linear(x1, x1)

        self.norm1 = nn.LayerNorm(x1)
        self.norm2 = nn.LayerNorm(x1) 
        
        init.constant_(self.final_linear.weight, 0)
        if self.final_linear.bias is not None:
            init.constant_(self.final_linear.bias, 0)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, tensor_A, tensor_B):
        batch_size = tensor_A.size(0)

        Q = self.split_heads(self.query(tensor_A), batch_size)
        K = self.split_heads(self.key(tensor_B), batch_size)
        V = self.split_heads(self.value(tensor_B), batch_size)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.depth ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_scores, V)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        output = attention_output.view(batch_size, -1, self.num_heads * self.depth)
        
        output = self.norm1(output + tensor_A)
        output = self.norm2(self.final_linear(output) + output)
        return output


def evenly_sample_or_duplicate_frames(video_tensor, target_frames=32):
    num_frames = video_tensor.size(0)
    if target_frames <= num_frames:
        indices = torch.linspace(0, num_frames - 1, steps=target_frames).long()
        return video_tensor[indices]
    else:
        scale_factor = target_frames / num_frames
        repeated_indices = (torch.arange(target_frames) / scale_factor).long()
        return video_tensor[repeated_indices]
                    
class LMModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        pattern_provider (CodebooksPatternProvider): Pattern provider for codebook interleaving.
        condition_provider (MusicConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        n_q (int): Number of parallel streams to model.
        card (int): Cardinality, vocabulary size.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_first (bool): Use pre-norm instead of post-norm.
        emb_lr (float, optional): Embedding-specific learning rate.
        bias_proj (bool): Use bias for output projections.
        weight_init (str, optional): Method for weight initialization.
        depthwise_init (str, optional): Method for depthwise weight initialization.
        zero_bias_init (bool): If true and bias in Linears, initialize bias to zeros.
        cfg_dropout (float): Classifier-free guidance dropout.
        cfg_coef (float): Classifier-free guidance coefficient.
        attribute_dropout (dict): Attribute dropout probabilities.
        two_step_cfg (bool): Whether to run classifier free-guidance with 2 distinct steps.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(self, pattern_provider: CodebooksPatternProvider, condition_provider: ConditioningProvider, 
                 visual_encoder,
                 if_add_gobal,
                 fuser: ConditionFuser, n_q: int = 8, card: int = 1024, dim: int = 128, num_heads: int = 8,
                 hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False,
                 depth=2,
                 temporal_dim=768,
                 dim_head=64,
                 **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.visual_encoder = visual_encoder
        self.if_add_gobal = if_add_gobal
        self.temporal_dim = temporal_dim
        
        self.fuser = fuser
        self.card = card
        embed_dim = self.card + 1
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        self.two_step_cfg = two_step_cfg
        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)
        
        
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None
        
        if self.visual_encoder == 'clip':
            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
                             
        else:
            print(f'the encoder now is:{self.visual_encoder}')
            print(f'please input the right video encoder.')
            exit()
        
        if self.visual_encoder == 'clip':
            temporal_dim = 768 
            self.local_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim))
            self.visual_encoder_model = self.visual_encoder_model.eval()
            for param in self.visual_encoder_model.parameters():
                param.requires_grad = False

        self.local_temporal_transformer = Transformer(temporal_dim, depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]

        if self.if_add_gobal:
            if self.visual_encoder == 'clip':
                self.global_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim))

            self.global_temporal_transformer = Transformer(temporal_dim, depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.) # [768, 4, 16, 64, 768*4]
            
            cross_attention_num_heads = 3 # MultiHeadCrossAttention
            self.multi_head_cross_attention = MultiHeadCrossAttention(temporal_dim, cross_attention_num_heads)
        
        self.visual_feature_proj = nn.Linear(temporal_dim, dim)                       


    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialization of the transformer module weights.

        Args:
            weight_init (str, optional): Weight initialization strategy. See `get_init_fn for valid options.
            depthwise_init (str, optional): Depthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)


    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def compute_video_emb(self, video_tensor_list: tp.List, device: str) -> torch.Tensor:
        assert isinstance(video_tensor_list, list)
        assert self.if_add_gobal
        assert len(video_tensor_list) == 2

        [local_video_tensor, global_video_tensor] = video_tensor_list
        local_image = local_video_tensor.to(dtype=torch.float32)
        global_image = global_video_tensor.to(dtype=torch.float32)

        # Frame is folded into batch dimension (no cross frame attention)
        local_batch_size, _, local_time_length, _, _ = local_image.size()
        local_image = einops.rearrange(local_image, 'b c t h w -> (b t) c h w')

        global_batch_size, _, global_time_length, _, _ = global_image.size()
        global_image = einops.rearrange(global_image, 'b c t h w -> (b t) c h w')

        local_temporal_transformer = self.local_temporal_transformer
        global_temporal_transformer = self.global_temporal_transformer

        local_video_inputs = self.processor(images=local_image.float(), return_tensors="pt")
        local_pixel_values = local_video_inputs['pixel_values'].to(device)

        global_video_inputs = self.processor(images=global_image.float(), return_tensors="pt")
        global_pixel_values = global_video_inputs['pixel_values'].to(device)

        #b: batch size 8 
        #t: number of frames (2fps * 29s = 58)
        #q: Number of patches + cls token 224/32 x 224/32 + 1 = 7 x 7 + 1 = 50
        #h: hidden size of the visual encoder each patch gets encoded to = 768
        #c: number of channels 3
        #ht: height of the video frame 224
        #w: width of the video frame 224

        #INPUT: local video tensors [b, c, t, ht, w]  (8, 3, 58, 224, 224)
        #FLatten for Batch Processing [b, c, t, ht, ww] -> [b*t, c, ht, w] (464, 3, 224, 224)
        #Encode video tensors using CLIP: [b*t, c, ht, w] -> [b*t, q, h] (464, 50, 768)
        #Add positional embedding to the encoded video tensors [b*t, q, h] + [1, 50, h] -> [b*t, q, h] 
        #Pass through the temporal transformer [b*t, q, h] -> [b*t, q, h] (Each patch of a frame attends to all other patches of the same frame)
        #Reshape to [b*t, q, h] -> [b, t, q, h] and then concatenate all patches along temporal dim [b, t*q, h] (8, 58, 50, 768) -> (8, 2900, 768)

        if self.visual_encoder == 'clip':
            with torch.no_grad():
                local_video_hidden = self.visual_encoder_model(pixel_values=local_pixel_values).last_hidden_state
            local_video_hidden += self.local_pos_embedding
            # this only runs spatial attention not temporal attention
            # this is just a second spatial attention head on top of CLIPs output (also spatial attention)
            local_video_hidden = local_temporal_transformer(local_video_hidden)
            local_video_hidden = einops.rearrange(
                local_video_hidden, '(b t) q h -> b (t q) h',
                b=local_batch_size, t=local_time_length
            )

            with torch.no_grad():
                global_video_hidden = self.visual_encoder_model(pixel_values=global_pixel_values).last_hidden_state
            global_video_hidden += self.global_pos_embedding
            global_video_hidden = global_temporal_transformer(global_video_hidden)
            global_video_hidden = einops.rearrange(
                global_video_hidden, '(b t) q h -> b (t q) h',
                b=global_batch_size, t=global_time_length
            )

        # Only here the spatiotemporal attention is applied
        # Each local video frame attends to all other global video frames
        # Outputs [b, t_local*q, h] where each of  t_local*q attends to all other t_global*q (8, 2900, 768)
        video_hidden = self.multi_head_cross_attention(local_video_hidden, global_video_hidden)

        # Linear Projection from [b, t_local*q, h] -> [b, t_local*q, dim] 8, 2900, 128)
        video_emb = self.visual_feature_proj(video_hidden)

        return video_emb


    def forward(self, sequence: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                video_tensor_list: tp.List,
                precomputed_video_emb: tp.Optional[torch.Tensor] = None  # 新增参数
                ) -> torch.Tensor:

        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        self.device = input_.device
        assert self.device != "cpu"

        if precomputed_video_emb is None:
            video_emb = self.compute_video_emb(video_tensor_list, device=self.device)
        else:
            video_emb = precomputed_video_emb

        out = self.transformer(input_, cross_attention_src=video_emb)
        if self.out_norm:
            out = self.out_norm(out)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)

        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]
        return logits  # [B, K, S, card]


    def compute_predictions(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors_list: tp.List) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list of ConditioningAttributes): Conditions to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as condition_tensors.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see conditions.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        B, K, T = codes.shape
        codes = codes.contiguous()
        
        assert isinstance(condition_tensors_list, list)
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )
        
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, conditions, condition_tensors_list)  # [B, K, S, card]


        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        return LMOutput(logits, logits_mask)


    def _sample_next_token(
            self,
            sequence: torch.Tensor,
            cfg_conditions_list: tp.List,
            unconditional_state: State,
            use_sampling: bool = False,
            temp: float = 1.0,
            top_k: int = 0,
            top_p: float = 0.0,
            cfg_coef: tp.Optional[float] = None,
            two_step_cfg: tp.Optional[bool] = None,
            precomputed_video_emb: tp.Optional[torch.Tensor] = None  # 新增参数
        ) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float, optional): classifier free guidance coefficient.
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg

        assert isinstance(cfg_conditions_list, list)
        assert len(cfg_conditions_list) == 2
        local_cfg_conditions = cfg_conditions_list[0]
        global_cfg_conditions = cfg_conditions_list[1]

        if two_step_cfg and local_cfg_conditions != {}:
            assert isinstance(local_cfg_conditions, tuple), type(local_cfg_conditions)
            local_condition_tensors, local_null_condition_tensors = local_cfg_conditions
            global_condition_tensors, global_null_condition_tensors = global_cfg_conditions
            cond_logits = model(sequence, conditions=[], condition_tensors=[local_condition_tensors, global_condition_tensors])

            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = model(sequence, conditions=[], condition_tensors=[local_null_condition_tensors, global_null_condition_tensors])
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        else:
            local_condition_tensors = cfg_conditions_list[0].to(sequence.device)
            global_condition_tensors = cfg_conditions_list[1].to(sequence.device)
            sequence = torch.cat([sequence, sequence], dim=0)
            
            if precomputed_video_emb is None:
                video_emb = self.compute_video_emb([cfg_conditions_list[0], cfg_conditions_list[1]], device=sequence.device)
            else:
                video_emb = precomputed_video_emb

            all_logits = model(
                sequence,
                conditions=[], 
                video_tensor_list=[],  
                precomputed_video_emb=video_emb  
            )
            cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token


    @torch.no_grad()
    def generate(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions_list: tp.List = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 two_step_cfg: tp.Optional[bool] = None,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (torch.Tensor, optional): Prompt tokens of shape [B, K, T].
            conditions_tensors (list of ConditioningAttributes, optional): List of conditions.
            num_samples (int, optional): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coeff (float, optional): Classifier-free guidance coefficient.
            two_step_cfg (bool, optional): Whether to perform classifier-free guidance with two steps generation.
            remove_prompts (bool): Whether to remove prompts from generation or not.
            check (bool): Whether to apply further checks on generated sequence.
            callback (Callback, optional): Callback function to report generation progress.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device
        assert isinstance(conditions_list, list)
        
        assert len(conditions_list) == 2
        local_conditions = conditions_list[0]
        global_conditions = conditions_list[1]
        # Checking all input shapes are consistent.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif local_conditions is not None:
            possible_num_samples.append(len(local_conditions))
        else:
            possible_num_samples.append(1)
            
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

        local_cfg_conditions: CFGConditions
        global_cfg_conditions: CFGConditions
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        local_null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(local_conditions)
        local_cfg_conditions = torch.cat((local_conditions, local_null_conditions), dim=0)
        global_null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(global_conditions)
        global_cfg_conditions = torch.cat((global_conditions, global_null_conditions), dim=0)

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        assert start_offset < max_gen_len

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1


        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        gen_codes[..., :start_offset] = prompt
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        video_emb = self.compute_video_emb([local_cfg_conditions, global_cfg_conditions], device=device)

        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]

            for offset in range(start_offset_sequence, gen_sequence_len):
                curr_sequence = gen_sequence[..., prev_offset:offset]
                curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                if check:
                    assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                    assert not (curr_sequence == unknown_token).any()
                next_token = self._sample_next_token(
                    curr_sequence, 
                    [local_cfg_conditions, global_cfg_conditions], 
                    unconditional_state, 
                    use_sampling, 
                    temp, 
                    top_k, 
                    top_p,
                    cfg_coef=cfg_coef, 
                    two_step_cfg=two_step_cfg,
                    precomputed_video_emb=video_emb  # 
                )
                valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                next_token[~valid_mask] = self.special_token_id
                gen_sequence[..., offset:offset+1] = torch.where(
                    gen_sequence[..., offset:offset+1] == unknown_token,
                    next_token, 
                    gen_sequence[..., offset:offset+1]
                )
                prev_offset = offset
                if callback is not None:
                    callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)

        unconditional_state.clear()
        assert not (gen_sequence == unknown_token).any()
        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes
