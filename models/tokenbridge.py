from functools import partial
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers import Mlp
from timm.models.vision_transformer import Block


def build_causal_mask(seq_length):
    """
    Create a causal attention mask for autoregressive prediction.
    
    Args:
        seq_length (int): Length of the sequence
        
    Returns:
        torch.Tensor: Causal attention mask with -inf in upper triangle
    """
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1) 
    return mask

def init_weights(module):
    """
    Initialize weights for different types of layers.
    
    Args:
        module: Neural network module to initialize
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)

def modulate(x, shift, scale):
    """
    Apply AdaLN modulation with shift and scale parameters.
    
    Args:
        x (torch.Tensor): Input tensor
        shift (torch.Tensor): Shift parameter
        scale (torch.Tensor): Scale parameter
        
    Returns:
        torch.Tensor: Modulated tensor
    """
    return x * (1 + scale) + shift

def mask_by_order(mask_len, order, bsz, seq_len):
    """
    Create a mask based on the specified order and mask length.
    
    Args:
        mask_len (int): Number of tokens to mask
        order (torch.Tensor): Order of token masking
        bsz (int): Batch size
        seq_len (int): Sequence length
        
    Returns:
        torch.Tensor: Boolean mask tensor
    """
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], 
                          src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class CausalAttention(nn.Module):
    """
    Causal self-attention module for autoregressive prediction with KV-caching support.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., 
                 proj_drop=0., norm_layer=nn.LayerNorm):
        """
        Initialize the causal attention module.
        
        Args:
            dim (int): Input dimension
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to use bias in QKV projection
            qk_norm (bool): Whether to apply normalization to Q and K
            attn_drop (float): Dropout rate for attention weights
            proj_drop (float): Dropout rate for projection output
            norm_layer: Normalization layer class
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_cache(self):
        """Reset the key and value caches."""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, attn_mask=None):
        """
        Forward pass for causal attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C]
            attn_mask (torch.Tensor): Optional attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape [B, N, C]
        """
        if attn_mask is None:
            attn_mask = build_causal_mask(x.size(1)).to(x.device)
            
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_cache:
            if self.k_cache is None and self.v_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
                k, v = self.k_cache, self.v_cache

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalBlock(nn.Module):
    """
    Transformer block with causal attention.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False,
                 proj_drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Initialize the causal block.
        
        Args:
            dim (int): Input dimension
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio for MLP hidden dimension
            qkv_bias (bool): Whether to use bias in QKV projection
            qk_norm (bool): Whether to apply normalization to Q and K
            proj_drop (float): Dropout rate for projections
            attn_drop (float): Dropout rate for attention weights
            act_layer: Activation layer class
            norm_layer: Normalization layer class
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, attn_mask=None, c=None):
        """
        Forward pass for the causal block.
        
        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor): Optional attention mask
            c (torch.Tensor): Conditioning tensor for AdaLN modulation
            
        Returns:
            torch.Tensor: Output tensor
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    Final normalization layer.
    """
    def __init__(self, dim, norm_layer):
        """
        Initialize the final layer.
        
        Args:
            dim (int): Input dimension
            norm_layer: Normalization layer class
        """
        super().__init__()
        self.norm_final = norm_layer(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2*dim)
        )
    
    def forward(self, x, c):
        """
        Forward pass for the final layer.
        
        Args:
            x (torch.Tensor): Input tensor
            c (torch.Tensor): Conditioning tensor for AdaLN modulation
            
        Returns:
            torch.Tensor: Output tensor
        """
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class TokenBridge(nn.Module):
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 channel_embed_dim=256, channel_depth=4, channel_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 quant_levels=32,  # 4bit = 16 levels
                 vae_embed_dim=16,   # VAE channels
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 grad_checkpointing=False,
                 quant_min=-5.0,
                 quant_max=5.0,
                 use_quantized_value=True,
                 std_range=3.0
                 ):
        """
        Initialize the TokenBridge model.
        
        Args:
            img_size (int): Input image size
            vae_stride (int): Stride of the VAE
            patch_size (int): Size of the patches
            encoder_embed_dim (int): Embedding dimension for the encoder
            encoder_depth (int): Number of encoder layers
            encoder_num_heads (int): Number of attention heads in encoder
            decoder_embed_dim (int): Embedding dimension for the decoder
            decoder_depth (int): Number of decoder layers
            decoder_num_heads (int): Number of attention heads in decoder
            channel_embed_dim (int): Embedding dimension for the channel-wise AR
            channel_depth (int): Number of layers in channel-wise AR
            channel_heads (int): Number of attention heads in channel-wise AR
            mlp_ratio (float): Ratio for MLP hidden dimension
            norm_layer: Normalization layer class
            quant_levels (int): Number of quantization levels
            vae_embed_dim (int): Dimension of VAE embeddings
            mask_ratio_min (float): Minimum mask ratio
            label_drop_prob (float): Probability to drop class labels
            class_num (int): Number of classes
            attn_dropout (float): Dropout rate for attention
            proj_dropout (float): Dropout rate for projections
            buffer_size (int): Size of the token buffer
            grad_checkpointing (bool): Whether to use gradient checkpointing
            quant_min (float): Minimum value for quantization
            quant_max (float): Maximum value for quantization
            use_quantized_value (bool): Whether to use quantized values
            std_range (float): Standard deviation range for quantization
        """
        super().__init__()
        # --------------------------------------------------------------------------
        # VAE and quantization specifics
        self.quant_levels = quant_levels
        self.vae_embed_dim = vae_embed_dim
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.std_range = std_range
        
        # --------------------------------------------------------------------------
        # Define optimal channel order based on frequency content
        self.channel_order = torch.tensor([
            8, 1, 10, 9, 6, 12, 4, 0, 3, 11, 7, 14, 2, 15, 13, 5
        ])
        # Define inverse mapping (for target mapping during training)
        self.inv_channel_order = torch.zeros_like(self.channel_order)
        for i, j in enumerate(self.channel_order):
            self.inv_channel_order[j] = i

            
        # --------------------------------------------------------------------------
        # Model architecture specifics
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing
        
        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        
        # --------------------------------------------------------------------------
        # MAR variant masking ratio
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        
        # --------------------------------------------------------------------------
        # Encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))
        
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer, proj_drop=proj_dropout,
                  attn_drop=attn_dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(encoder_embed_dim)
        
        # --------------------------------------------------------------------------
        # Decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer, proj_drop=proj_dropout,
                  attn_drop=attn_dropout)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.output_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))
        
        # --------------------------------------------------------------------------
        # Channel AR components
        self.condition_proj = nn.Linear(decoder_embed_dim, channel_embed_dim)
        self.channel_embed = nn.Parameter(torch.zeros(1, vae_embed_dim, channel_embed_dim))
        self.timesteps_embeddings = nn.Parameter(torch.zeros(1, vae_embed_dim, channel_embed_dim))
        
        # Token embeddings (arranged in optimized order)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(quant_levels, channel_embed_dim)
            for _ in range(vae_embed_dim-1)
        ])
        
        # Channel transformer blocks
        self.channel_blocks = nn.ModuleList([
            CausalBlock(
                dim=channel_embed_dim,
                num_heads=channel_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=True,
                proj_drop=proj_dropout,
                attn_drop=attn_dropout,
                norm_layer=norm_layer
            ) for _ in range(channel_depth)
        ])
        self.channel_norm = norm_layer(channel_embed_dim)
        
        self.channel_final = FinalLayer(channel_embed_dim, norm_layer=norm_layer)
        
        # Channel prediction heads (arranged in optimized order)
        self.channel_heads = nn.ModuleList([
            nn.Linear(channel_embed_dim, quant_levels)
            for _ in range(vae_embed_dim)
        ])
        
        # Create channel causal mask
        channel_mask = torch.empty(vae_embed_dim, vae_embed_dim)
        channel_mask.fill_(float("-inf"))
        channel_mask.triu_(1)
        self.register_buffer('channel_mask', channel_mask)
        
        self.initialize_weights()
        self._init_gaussian_quantization()

        
    
    def _init_gaussian_quantization(self):
        """
        Initialize Gaussian-based quantization boundaries and reconstruction values.
        These are used for the post-training quantization process.
        """
        device = next(self.parameters()).device
        dtype = torch.float32
        
        probs = torch.linspace(0, 1, self.quant_levels + 1, device=device, dtype=dtype)
        boundaries = torch.tensor(stats.norm.ppf(probs.cpu()), device=device, dtype=dtype)
        boundaries = torch.clamp(boundaries, -self.std_range, self.std_range)
        
        reconstruction_values = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i+1]
            mean = self._truncated_normal_mean(a, b)
            reconstruction_values.append(mean)
            
        self.register_buffer('reconstruction_values', 
                           torch.tensor(reconstruction_values, dtype=dtype))
        self.register_buffer('boundaries', boundaries)

    def _truncated_normal_mean(self, a, b):
        """
        Calculate expected value of the truncated normal distribution between a and b.
        
        Args:
            a (torch.Tensor): Lower bound
            b (torch.Tensor): Upper bound
            
        Returns:
            torch.Tensor: Expected value
        """
        sqrt_2 = math.sqrt(2)
        sqrt_2pi = math.sqrt(2 * math.pi)
        
        phi_a = torch.exp(-0.5 * a**2) / sqrt_2pi
        phi_b = torch.exp(-0.5 * b**2) / sqrt_2pi
        
        Phi_a = 0.5 * (1 + torch.erf(a / sqrt_2))
        Phi_b = 0.5 * (1 + torch.erf(b / sqrt_2))
        
        denominator = Phi_b - Phi_a
        denominator = torch.where(denominator == 0, 
                                torch.tensor(1e-10, device=a.device, dtype=a.dtype), 
                                denominator)
        
        return (phi_a - phi_b) / denominator
    
    def initialize_weights(self):
        """
        Initialize model weights with appropriate distributions.
        """
        # Initialize embeddings
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.channel_embed, std=.02)
        torch.nn.init.normal_(self.timesteps_embeddings, std=.02)
        
        # Initialize position embeddings
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.output_pos_embed, std=.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
        for block in self.channel_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.channel_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.channel_final.adaLN_modulation[-1].bias, 0)
    
    def _init_weights(self, m):
        """
        Weight initialization helper function for network modules.
        
        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


    def quantize(self, x):
        """
        Apply dimension-wise quantization to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor to quantize
            
        Returns:
            tuple: Quantized indices and their corresponding dequantized values
        """
        x_normalized = (x - self.quant_min) / (self.quant_max - self.quant_min) * \
                      (2 * self.std_range) - self.std_range
        x_clamped = x_normalized.clamp(-self.std_range, self.std_range)
        
        x_expanded = x_clamped.unsqueeze(-1)
        dists = (x_expanded - self.reconstruction_values).abs()
        indices = dists.argmin(dim=-1)
        
        normalized_values = self.reconstruction_values
        values = (normalized_values + self.std_range) / (2 * self.std_range) * \
                (self.quant_max - self.quant_min) + self.quant_min
        dequant = values[indices]
        
        return indices, dequant

    def patchify(self, x):
        """
        Convert images into patches.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Patched tensor of shape [B, H*W, C*P*P]
        """
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        """
        Convert patches back to images.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, H*W, C*P*P]
            
        Returns:
            torch.Tensor: Unpatchified tensor of shape [B, C, H, W]
        """
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def sample_orders(self, bsz):
        """
        Sample random ordering for token prediction.
        
        Args:
            bsz (int): Batch size
            
        Returns:
            torch.Tensor: Random order indices
        """
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders
    
    def random_masking(self, x, orders):
        """
        Create random masks for masked autoencoding.
        
        Args:
            x (torch.Tensor): Input tensor
            orders (torch.Tensor): Token ordering
            
        Returns:
            torch.Tensor: Mask tensor
        """
        bsz = x.size(0)
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(self.seq_len * mask_rate))
        mask = torch.zeros(bsz, self.seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                           src=torch.ones(bsz, self.seq_len, device=x.device))
        return mask
    
    def forward_mae_encoder(self, x, mask, class_embedding):
        """
        Forward pass through the masked autoencoder encoder.
        
        Args:
            x (torch.Tensor): Input features
            mask (torch.Tensor): Masking tensor
            class_embedding (torch.Tensor): Class embedding
            
        Returns:
            torch.Tensor: Encoded tensor
        """
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        
        # concat buffer
        x = torch.cat([
            torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device),
            x
        ], dim=1)
        mask_with_buffer = torch.cat([
            torch.zeros(x.size(0), self.buffer_size, device=x.device),
            mask
        ], dim=1)
        
        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + \
                            (1 - drop_latent_mask) * class_embedding
        
        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        
        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)
        
        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)
        
        return x

    def forward_mae_decoder(self, x, mask):
        """
        Forward pass through the masked autoencoder decoder.
        
        Args:
            x (torch.Tensor): Input features from encoder
            mask (torch.Tensor): Masking tensor
            
        Returns:
            torch.Tensor: Decoded tensor
        """
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([
            torch.zeros(x.size(0), self.buffer_size, device=x.device),
            mask
        ], dim=1)
        
        # pad mask tokens
        mask_tokens = self.mask_token.repeat(
            mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = \
            x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        
        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned
        
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)
        
        x = x[:, self.buffer_size:]
        x = x + self.output_pos_embed
        return x

    def forward(self, imgs, labels):
        """
        Forward pass for training.
        
        Args:
            imgs (torch.Tensor): Input images
            labels (torch.Tensor): Class labels
            
        Returns:
            torch.Tensor: Loss value
        """
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        
        # Use expectation-based Gaussian quantization
        quant_indices, quant_values = self.quantize(gt_latents)
        
        # Choose whether to use original values or quantized values
        encoder_input = quant_values
        
        # class embed
        class_embedding = self.class_emb(labels)
        
        # generate mask
        orders = self.sample_orders(bsz=x.size(0))
        gt_tokens = quant_indices[..., self.channel_order]  # Reorder channels before passing to AR
        mask = self.random_masking(encoder_input, orders)
        
        # encoder
        x = self.forward_mae_encoder(encoder_input, mask, class_embedding)
        
        # decoder
        z = self.forward_mae_decoder(x, mask)
        
        # channel AR prediction
        logits = self.forward_channel_ar(z, gt_tokens, mask)
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.quant_levels),
            gt_tokens[mask.bool()].reshape(-1).long()
        )
        
        return loss

    def enable_cache(self):
        """Enable KV-caching for efficient autoregressive generation."""
        for block in self.channel_blocks:
            if hasattr(block, 'attn'):
                block.attn.use_cache = True
                block.attn.reset_cache()
            
    def disable_cache(self):
        """Disable KV-caching."""
        for block in self.channel_blocks:
            if hasattr(block, 'attn'):
                block.attn.use_cache = False
                block.attn.reset_cache()
    
    def forward_channel_ar(self, spatial_features, channel_targets, mask):
        """
        Forward pass for the channel-wise autoregressive predictor.
        
        Args:
            spatial_features (torch.Tensor): Spatial features from decoder
            channel_targets (torch.Tensor): Target tokens for channel prediction
            mask (torch.Tensor): Mask tensor
            
        Returns:
            torch.Tensor: Prediction logits
        """
        # Extract masked positions
        mask_indices = mask.bool()
        spatial_features = spatial_features[mask_indices]  # [num_masked, decoder_embed_dim]
        channel_targets = channel_targets[mask_indices]    # [num_masked, C]
        
        # Project decoder features
        cond = self.condition_proj(spatial_features)  # [num_masked, channel_embed_dim]
        
        # Initialize sequence with condition
        channel_sequence = [cond.unsqueeze(1)]  # [num_masked, 1, channel_embed_dim]
        
        # Add embeddings for previous channels
        for i in range(self.vae_embed_dim-1):
            channel_embed = self.token_embeddings[i](channel_targets[..., i])
            channel_sequence.append(channel_embed.unsqueeze(1))
        
        # Stack sequence
        x = torch.cat(channel_sequence, dim=1)  # [num_masked, C, channel_embed_dim]
        
        # Add position embedding and apply transformer
        x = x + self.channel_embed
        cond = cond.unsqueeze(1) + self.timesteps_embeddings[:, :cond.shape[1]]
        for block in self.channel_blocks:
            x = block(x, attn_mask=self.channel_mask, c=cond)
        x = self.channel_norm(x)
        
        x = self.channel_final(x, cond)
        
        # Get predictions
        logits = torch.stack([
            head(x[:, i]) for i, head in enumerate(self.channel_heads)
        ], dim=1)  # [num_masked, C, quant_levels]
        
        return logits
    
    def _sample_channel_ar(self, spatial_features, temperature):
        """
        Autoregressive sampling of channel tokens.
        
        Args:
            spatial_features (torch.Tensor): Spatial features for conditioning
            temperature (float): Sampling temperature
            
        Returns:
            torch.Tensor: Predicted tokens
        """
        B = spatial_features.shape[0]
        
        # Project condition
        cond = self.condition_proj(spatial_features.squeeze(1))  # [B, channel_embed_dim]
        
        # Broadcast to full sequence length
        cond = cond.unsqueeze(1) + self.timesteps_embeddings  # [B, vae_embed_dim, channel_embed_dim]

        # Initialize sequence with condition
        channel_sequence = [cond[:, :1]]  # Take first timestep
        
        next_tokens = []
        
        # Autoregressively predict each channel
        for c in range(self.vae_embed_dim):
            # Stack sequence
            x = torch.cat(channel_sequence, dim=1)  # [B, curr_len, channel_embed_dim]
            x = x + self.channel_embed[:, :x.shape[1]]
            
            curr_len = x.size(1)
            attn_mask = self.channel_mask[:curr_len, :curr_len]
            
            # Use current timestep's condition
            curr_cond = cond[:, :curr_len]
            
            for block in self.channel_blocks:
                x = block(x, attn_mask=attn_mask, c=curr_cond)
            x = self.channel_norm(x)
            
            x = self.channel_final(x, curr_cond)
            
            # Get prediction for current channel
            logits = self.channel_heads[c](x[:, -1])
            
            # Sample token
            if temperature == 0:
                token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            next_tokens.append(token)
            
            # Add token embedding to sequence if not last channel
            if c < self.vae_embed_dim - 1:
                token_embed = self.token_embeddings[c](token)
                channel_sequence.append(token_embed.unsqueeze(1))
        
        return torch.stack(next_tokens, dim=1)
    

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0):
        """
        Sample tokens using iterative autoregressive prediction.
        
        Args:
            bsz (int): Batch size
            num_iter (int): Number of sampling iterations
            cfg (float): Classifier-free guidance scale
            cfg_schedule (str): Schedule for classifier-free guidance
            labels (torch.Tensor): Optional class labels
            temperature (float): Sampling temperature
            
        Returns:
            torch.Tensor: Generated continuous tokens
        """
        device = next(self.parameters()).device
        
        # Initialize with original order
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.vae_embed_dim).long().cuda()
        orders = self.sample_orders(bsz)
        
        for step in range(num_iter):
            cur_tokens = tokens.clone()
            
            # Calculate reconstruction values directly from tokens in original order
            cur_values = self.reconstruction_values[cur_tokens]
            cur_continuous = (cur_values + self.std_range) / (2 * self.std_range) * \
                        (self.quant_max - self.quant_min) + self.quant_min
            
            if labels is not None:
                cond_embedding = self.class_emb(labels)
            else:
                cond_embedding = self.fake_latent.repeat(bsz, 1)
            uncond_embedding = self.fake_latent.repeat(bsz, 1)
            
            if cfg != 1.0:
                x_cond = self.forward_mae_encoder(cur_continuous, mask, cond_embedding)
                z_cond = self.forward_mae_decoder(x_cond, mask)
                
                x_uncond = self.forward_mae_encoder(cur_continuous, mask, uncond_embedding)
                z_uncond = self.forward_mae_decoder(x_uncond, mask)
            else:
                x = self.forward_mae_encoder(cur_continuous, mask, cond_embedding)
                z = self.forward_mae_decoder(x, mask)
            
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len)
            )
            
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask.bool()
            else:
                mask_to_pred = torch.logical_xor(mask.bool(), mask_next.bool())
            
            # AR prediction part
            if cfg != 1.0:
                z_cond = z_cond[mask_to_pred.nonzero(as_tuple=True)]
                z_uncond = z_uncond[mask_to_pred.nonzero(as_tuple=True)]
                
                if cfg_schedule == "linear":
                    cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
                else:
                    cfg_iter = cfg
                    
                guided_feature = z_uncond + cfg_iter * (z_cond - z_uncond)
                next_tokens = self._sample_channel_ar(guided_feature.unsqueeze(1), temperature)  # Returns tokens in new order
                # Convert back to original order for storage
                next_tokens = next_tokens[..., self.inv_channel_order]
            else:
                z = z[mask_to_pred.nonzero(as_tuple=True)]
                next_tokens = self._sample_channel_ar(z.unsqueeze(1), temperature)  # Returns tokens in new order
                # Convert back to original order for storage
                next_tokens = next_tokens[..., self.inv_channel_order]
                
            # Update tokens in original order
            mask_indices = mask_to_pred.nonzero(as_tuple=True)
            cur_tokens[mask_indices] = next_tokens
            tokens = cur_tokens
            mask = mask_next
        
        # Final reconstruction (already in original order)
        continuous_values = self.reconstruction_values[tokens]
        continuous_tokens = (continuous_values + self.std_range) / (2 * self.std_range) * \
                        (self.quant_max - self.quant_min) + self.quant_min
        continuous_tokens = self.unpatchify(continuous_tokens)
        
        return continuous_tokens

def tokenbridge_base(**kwargs):
    model = TokenBridge(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        channel_embed_dim=768, channel_depth=4, channel_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def tokenbridge_large(**kwargs):
    model = TokenBridge(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        channel_embed_dim=1024, channel_depth=4, channel_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def tokenbridge_huge(**kwargs):
    model = TokenBridge(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        channel_embed_dim=1024, channel_depth=6, channel_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
