"""
Modified from Audiocraft (https://github.com/facebookresearch/audiocraft)
This module defines the language model (LM) code with transformer, video conditioning,
and various sampling methods.
"""

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

# Type aliases for conditioning tensors
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """
    Returns a partial function for initializing a layer's weight tensor
    based on the specified method and initialization depth.

    Args:
        method (str): The initialization method ("gaussian" or "uniform").
        input_dim (int): Input dimension of the layer.
        init_depth (int, optional): Depth scaling factor for the initialization.

    Returns:
        A partial function that can be used as an initialization function.
    """
    std = 1 / math.sqrt(input_dim)
    # Rescale the standard deviation by the depth if provided.
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        # Truncated normal initialization with bounds at ±3*std
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        # Uniform initialization with bound that ensures std is as computed
        bound = math.sqrt(3) * std
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module, method: str, init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    """
    Initialize the given module (nn.Linear or nn.Embedding) using a chosen method.

    Args:
        m (nn.Module): Module to be initialized.
        method (str): Initialization method to use ("gaussian" or "uniform").
        init_depth (int, optional): Depth scaling factor if needed.
        zero_bias_init (bool): If True, initialize bias to zeros if the bias exists.

    Returns:
        None. The module's parameters are modified in-place.
    """
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        # Handle potential float16 weights on CPU
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
    """
    An embedding layer that allows a custom learning rate override.
    """
    def __init__(self, *args, lr=None, **kwargs):
        """
        Initialize a ScaledEmbedding.

        Args:
            *args, **kwargs: Arguments passed to nn.Embedding.
            lr (float, optional): Custom learning rate to be used for this embedding.
        """
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        """
        Prepares an optimizer parameter group for this embedding.

        Returns:
            dict: A parameter group possibly including a custom learning rate.
        """
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    """
    Dataclass used to store language model outputs.

    Attributes:
        logits (torch.Tensor): Output logits of shape [B, K, T, card] where:
            B - Batch size, K - Number of streams/codebooks, T - Timesteps, card - vocabulary cardinality.
        mask (torch.Tensor): Mask tensor of shape [B, K, T] indicating valid prediction positions.
    """
    logits: torch.Tensor
    mask: torch.Tensor


class Transformer(nn.Module):
    """
    A simple transformer block that applies a series of attention and feed-forward layers.

    Args:
        dim (int): Dimension of the input and output.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension for each attention head.
        mlp_dim (int): Dimension of the feed forward layer.
        dropout (float, optional): Dropout probability.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        # Create a list of transformer layers; each layer consists of attention and feed-forward sublayers.
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Process the input tensor through the transformer layers.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, sequence_length, dim].

        Returns:
            torch.Tensor: Processed tensor after transformer layers and a final normalization.
        """
        for attn, ff in self.layers:
            x = attn(x) + x  # Residual connection for attention output.
            x = ff(x) + x    # Residual connection for feed-forward output.
        return self.norm(x)  # Final layer normalization.


class MultiHeadCrossAttention(nn.Module):
    """
    Performs cross attention between two tensors using multiple attention heads.

    Given two input tensors, one is used to generate queries while the other provides keys and values.

    Args:
        x1 (int): The total dimension of each input.
        num_heads (int): The number of attention heads.
    """
    def __init__(self, x1, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Dimension per head.
        self.depth = x1 // num_heads

        # Linear layers for query, key, and value.
        self.query = nn.Linear(x1, x1)
        self.key = nn.Linear(x1, x1)
        self.value = nn.Linear(x1, x1)

        # Output projection layer.
        self.final_linear = nn.Linear(x1, x1)

        # Two normalization layers.
        self.norm1 = nn.LayerNorm(x1)
        self.norm2 = nn.LayerNorm(x1)
        
        # Initialize the final linear layer weights and biases to zero.
        init.constant_(self.final_linear.weight, 0)
        if self.final_linear.bias is not None:
            init.constant_(self.final_linear.bias, 0)
    
    def split_heads(self, x, batch_size):
        """
        Splits the input tensor into multiple heads.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_length, x1].
            batch_size (int): The batch size.

        Returns:
            torch.Tensor: Reshaped tensor of shape [batch, num_heads, seq_length, depth].
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, tensor_A, tensor_B):
        """
        Computes the cross-attention between tensor_A and tensor_B.

        Args:
            tensor_A (torch.Tensor): Tensor used for query generation. Shape [batch, seq_A, x1].
            tensor_B (torch.Tensor): Tensor used to generate keys and values. Shape [batch, seq_B, x1].

        Returns:
            torch.Tensor: Output tensor after cross attention. Shape [batch, seq_A, x1].
        """
        batch_size = tensor_A.size(0)

        # Generate queries, keys, and values and split them into heads.
        Q = self.split_heads(self.query(tensor_A), batch_size)
        K = self.split_heads(self.key(tensor_B), batch_size)
        V = self.split_heads(self.value(tensor_B), batch_size)

        # Compute scaled dot-product attention.
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.depth ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_scores, V)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        # Concatenate the heads.
        output = attention_output.view(batch_size, -1, self.num_heads * self.depth)
        
        # Apply residual connection and normalization.
        output = self.norm1(output + tensor_A)
        output = self.norm2(self.final_linear(output) + output)
        return output


def evenly_sample_or_duplicate_frames(video_tensor, target_frames=32):
    """
    Resamples or duplicates frames from a video tensor so that it contains the target number of frames.

    Args:
        video_tensor (torch.Tensor): Video frames tensor of shape [num_frames, ...].
        target_frames (int, optional): The desired number of frames.

    Returns:
        torch.Tensor: A tensor with exactly target_frames frames.
    """
    num_frames = video_tensor.size(0)
    if target_frames <= num_frames:
        # Evenly sample indices if there are more frames than required.
        indices = torch.linspace(0, num_frames - 1, steps=target_frames).long()
        return video_tensor[indices]
    else:
        # Duplicate frames if there are fewer frames than needed.
        scale_factor = target_frames / num_frames
        repeated_indices = (torch.arange(target_frames) / scale_factor).long()
        return video_tensor[repeated_indices]
                    
                    
class LMModel(StreamingModule):
    """
    Transformer-based language model that processes multiple streams of codes
    and incorporates video conditioning. Supports methods for sampling and generation
    using classifier-free guidance, cross attention, and streaming inference.

    Attributes:
        pattern_provider (CodebooksPatternProvider): Provides interleaving patterns for codes.
        condition_provider (ConditioningProvider): Supplies conditioning metadata.
        visual_encoder (str): Identifier for the visual encoder (e.g., 'clip').
        if_add_gobal (bool): Whether to use a global conditioning branch.
        fuser (ConditionFuser): Module to fuse conditioning information with LM input.
        n_q (int): Number of parallel code streams.
        card (int): Vocabulary size (cardinality).
        dim (int): Transformer encoder dimension.
        num_heads (int): Number of attention heads.
        hidden_scale (int): Multiplicative factor for hidden dimensions.
        ... (other parameters related to dropout, weight initialization, etc.)
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
        """
        Initialize the LMModel with a transformer, visual encoder, and various conditioning modules.

        Args:
            pattern_provider (CodebooksPatternProvider): Pattern provider for interleaving.
            condition_provider (ConditioningProvider): Provides conditioning info.
            visual_encoder (str): Identifier for video/visual encoder ('clip' supported).
            if_add_gobal (bool): Whether to include a global conditioning branch.
            fuser (ConditionFuser): Module to fuse conditioning into LM input.
            n_q (int): Number of code streams.
            card (int): Codebook (vocabulary) size.
            dim (int): Transformer dimension.
            num_heads (int): Number of attention heads.
            hidden_scale (int): Scale factor for feed-forward network dimension.
            norm (str): Normalization method.
            norm_first (bool): Whether to use pre-norm architecture.
            emb_lr (float, optional): Learning rate for embeddings.
            bias_proj (bool): Whether to use bias in final projection layers.
            weight_init (str, optional): Weight initialization method.
            depthwise_init (str, optional): Method to use depth-based initialization.
            zero_bias_init (bool): If True, initialize biases as zeros.
            cfg_dropout (float): Dropout probability for classifier-free guidance.
            cfg_coef (float): Coefficient for classifier-free guidance.
            attribute_dropout (dict): Probabilities for attribute dropout.
            two_step_cfg (bool): Whether to perform two-step classifier-free guidance.
            depth (int): Depth for temporal transformer layers.
            temporal_dim (int): Dimension for temporal features.
            dim_head (int): Dimension per head for temporal transformers.
            **kwargs: Additional transformer parameters.
        """
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
        # Create multiple scaled embeddings for each code stream.
        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        # Initialize the streaming transformer for sequence processing.
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)
        
        # Create an optional normalization layer for output if using pre-norm architecture.
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        # Create linear layers to project transformer outputs to vocabulary logits.
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
        # Initialize weights based on the provided strategies.
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None
        
        # Set up the visual encoder. Currently supports 'clip'.
        if self.visual_encoder == 'clip':
            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            print(f'the encoder now is:{self.visual_encoder}')
            print(f'please input the right video encoder.')
            exit()
        
        # For CLIP, use fixed temporal dimension and position embeddings.
        if self.visual_encoder == 'clip':
            temporal_dim = 768 
            self.local_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim))
            # Freeze the visual encoder parameters.
            self.visual_encoder_model = self.visual_encoder_model.eval()
            for param in self.visual_encoder_model.parameters():
                param.requires_grad = False

        # Initialize local transformer for processing temporal features locally.
        self.local_temporal_transformer = Transformer(temporal_dim, depth, num_heads, dim_head, temporal_dim * hidden_scale, 0.)

        # If global conditioning is enabled, initialize extra global modules.
        if self.if_add_gobal:
            if self.visual_encoder == 'clip':
                self.global_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim))
            self.global_temporal_transformer = Transformer(temporal_dim, depth, num_heads, dim_head, temporal_dim * hidden_scale, 0.)
            cross_attention_num_heads = 3  # For cross attention between local and global features.
            self.multi_head_cross_attention = MultiHeadCrossAttention(temporal_dim, cross_attention_num_heads)
        
        # Final projection from video feature dimension to transformer dimension.
        self.visual_feature_proj = nn.Linear(temporal_dim, dim)

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """
        Initialize weights for embeddings, transformer layers and output linear layers based on provided methods.

        Args:
            weight_init (str, optional): Weight initialization method.
            depthwise_init (str, optional): Specifies depth-based initialization ('current' or 'global').
            zero_bias_init (bool): Whether to set biases to zero.
        
        Returns:
            None. Initializes weights of modules in-place.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        # Initialize each embedding layer.
        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        # Initialize each transformer layer with depth-based scaling.
        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        # Initialize each linear projection layer.
        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    @property
    def special_token_id(self) -> int:
        """
        Returns:
            int: The special token ID used to indicate tokens outside the vocabulary. 
                 It is set as the same as the vocabulary cardinality.
        """
        return self.card

    @property
    def num_codebooks(self) -> int:
        """
        Returns:
            int: The number of parallel code streams (codebooks).
        """
        return self.n_q

    def compute_video_emb(self, video_tensor_list: tp.List, device: str) -> torch.Tensor:
        """
        Compute a video embedding from a list of video tensors using the visual encoder and temporal transformers.

        Args:
            video_tensor_list (list): A list containing two video tensors:
                [local_video_tensor, global_video_tensor], each with shape [B, C, T, H, W].
            device (str): The torch device to place the tensors on.

        Returns:
            torch.Tensor: The computed video embeddings with shape [B, new_seq_length, dim].
        """
        # Verify input is a list and has exactly two elements.
        assert isinstance(video_tensor_list, list)
        assert self.if_add_gobal
        assert len(video_tensor_list) == 2

        [local_video_tensor, global_video_tensor] = video_tensor_list
        local_image = local_video_tensor.to(dtype=torch.float32)
        global_image = global_video_tensor.to(dtype=torch.float32)

        # Rearrange local and global image tensors to merge batch and time dimensions.
        # Flattening all the frames across the batch into a single dimension.
        local_batch_size, _, local_time_length, _, _ = local_image.size()
        local_image = einops.rearrange(local_image, 'b c t h w -> (b t) c h w')

        global_batch_size, _, global_time_length, _, _ = global_image.size()
        global_image = einops.rearrange(global_image, 'b c t h w -> (b t) c h w')

        local_temporal_transformer = self.local_temporal_transformer
        global_temporal_transformer = self.global_temporal_transformer

        # Process the local video frames using the processor.
        local_video_inputs = self.processor(images=local_image.float(), return_tensors="pt")
        local_pixel_values = local_video_inputs['pixel_values'].to(device)

        # Process the global video frames.
        global_video_inputs = self.processor(images=global_image.float(), return_tensors="pt")
        global_pixel_values = global_video_inputs['pixel_values'].to(device)

        # Use the visual encoder to extract features and add positional embeddings.
        if self.visual_encoder == 'clip':
            with torch.no_grad():
                local_video_hidden = self.visual_encoder_model(pixel_values=local_pixel_values).last_hidden_state
            local_video_hidden += self.local_pos_embedding
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

        # Fuse local and global video features using cross attention.
        video_hidden = self.multi_head_cross_attention(local_video_hidden, global_video_hidden)
        video_emb = self.visual_feature_proj(video_hidden)

        return video_emb

    def forward(self, sequence: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                video_tensor_list: tp.List,
                precomputed_video_emb: tp.Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass of the language model.

        Args:
            sequence (torch.Tensor): Input code tensor of shape [B, K, S] where
                B: batch size, K: number of codebooks, S: sequence length.
            conditions (list): List of conditioning attributes.
            video_tensor_list (list): List of video tensors used to compute video embeddings.
            precomputed_video_emb (torch.Tensor, optional): If provided, directly use these precomputed embeddings.

        Returns:
            torch.Tensor: Logits tensor of shape [B, K, S, card].
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        # Sum the embeddings across the codebooks.
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        self.device = input_.device
        assert self.device != "cpu"

        # Compute video embeddings if not precomputed.
        if precomputed_video_emb is None:
            video_emb = self.compute_video_emb(video_tensor_list, device=self.device)
        else:
            video_emb = precomputed_video_emb

        # Pass the summed embeddings and video conditioning into the transformer.
        out = self.transformer(input_, cross_attention_src=video_emb)
        if self.out_norm:
            out = self.out_norm(out)
        # Project transformer outputs to vocabulary logits for each codebook.
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)

        # If certain fuse conditions require, trim the logits.
        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]
        return logits  # [B, K, S, card]

    def compute_predictions(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors_list: tp.List) -> LMOutput:
        """
        Given input codes and conditions, computes predictions by applying an interleaving pattern,
        running the transformer, and reverting the pattern.

        Args:
            codes (torch.Tensor): Tensor of input codes with shape [B, K, T].
            conditions (list): List of conditioning attributes.
            condition_tensors_list (list): List of precomputed conditioning tensors.

        Returns:
            LMOutput: A dataclass with logits of shape [B, K, T, card] and a corresponding mask [B, K, T].
        """
        B, K, T = codes.shape
        codes = codes.contiguous()
        
        assert isinstance(condition_tensors_list, list)
        # Build sequence pattern according to interleaving strategy.
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )
        
        # Use the model to compute logits.
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, conditions, condition_tensors_list)  # [B, K, S, card]

        # Revert the pattern to recover original positions.
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
            precomputed_video_emb: tp.Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Sample the next token given a partial sequence and conditioning.

        Supports both greedy and sampling strategies (including top-k and top-p).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S].
            cfg_conditions_list (list): List containing local and global CFG conditions.
            unconditional_state (State): The streaming state for unconditional generation.
            use_sampling (bool): Whether to use sampling.
            temp (float): Temperature for sampling.
            top_k (int): K value for top-k sampling.
            top_p (float): P threshold for top-p sampling.
            cfg_coef (float, optional): Classifier-free guidance coefficient override.
            two_step_cfg (bool, optional): Whether to apply two-step CFG.
            precomputed_video_emb (torch.Tensor, optional): Precomputed video embeddings.

        Returns:
            torch.Tensor: Tensor of next token indices with shape [B, K, 1].
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
            # In two-step CFG: first compute conditional logits,
            # then swap state to compute unconditional logits.
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
            # Duplicate the sequence for conditional and unconditional branches.
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

        # Rearrange the logits to extract probabilities for the last generated token.
        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B, K, card]

        # If sampling is enabled, perform softmax sampling, otherwise pick the most likely token.
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
        """
        Generates token sequences from the model given an optional prompt and conditions.
        
        The generation is performed iteratively, sampling one token at a time until max_gen_len is reached.
        Supports classifier-free guidance and various sampling methods.

        Args:
            prompt (torch.Tensor, optional): Initial tokens of shape [B, K, T] serving as the prompt.
            conditions_list (list): List containing two sets of conditioning attributes (local and global).
            num_samples (int, optional): Number of samples to generate when prompt is None.
            max_gen_len (int): Maximum length of generated sequence.
            use_sampling (bool): Whether to use sampling (if False, use greedy sampling).
            temp (float): Temperature scaling for sampling.
            top_k (int): Top-k parameter for sampling.
            top_p (float): Top-p (nucleus) parameter for sampling.
            cfg_coef (float, optional): Classifier-free guidance coefficient.
            two_step_cfg (bool, optional): Whether to use two-step classifier-free guidance.
            remove_prompts (bool): If True, remove the prompt tokens from the final output.
            check (bool): If True, perform extra checks on the generated sequence.
            callback (callable, optional): A function to report generation progress (step, total steps).

        Returns:
            torch.Tensor: The generated tokens with shape [B, K, L] where L <= max_gen_len.
        """
        assert not self.training, "Generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device
        assert isinstance(conditions_list, list)
        
        # Ensure conditions_list has exactly 2 entries: local and global conditions.
        assert len(conditions_list) == 2
        local_conditions = conditions_list[0]
        global_conditions = conditions_list[1]
        # Determine how many samples to generate.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif local_conditions is not None:
            possible_num_samples.append(len(local_conditions))
        else:
            possible_num_samples.append(1)
            
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent input shapes"
        num_samples = possible_num_samples[0]

        # Prepare classifier-free guidance conditions (concatenate condition and null versions).
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

        # Build the sequence pattern used for generation.
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        unknown_token = -1  # This token indicates positions that are not generated yet.

        # Initialize a tensor for generated codes with unknown tokens.
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        gen_codes[..., :start_offset] = prompt
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        # Compute video embedding for the CFG conditions.
        video_emb = self.compute_video_emb([local_cfg_conditions, global_cfg_conditions], device=device)

        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]

            # Iteratively sample one token at a time.
            for offset in range(start_offset_sequence, gen_sequence_len):
                curr_sequence = gen_sequence[..., prev_offset:offset]
                curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                if check:
                    # Sanity checks: ensure that current sequence matches the mask.
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
                    precomputed_video_emb=video_emb
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
        # Final checks to ensure no unknown tokens remain and masking is correct.
        assert not (gen_sequence == unknown_token).any()
        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

        # Ensure that the final codes are valid (between 0 and vocabulary size).
        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes