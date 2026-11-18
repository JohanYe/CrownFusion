import math

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch import einsum, nn
from pc_encoder import Encoder as PC_Encoder


##################################################################################################
# SiT Model
##################################################################################################


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################


def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def SiT_L_1(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)


def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def SiT_B_1(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)


def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    "SiT-XL/2": SiT_XL_2,
    "SiT-XL/4": SiT_XL_4,
    "SiT-XL/8": SiT_XL_8,
    "SiT-L/2": SiT_L_2,
    "SiT-L/4": SiT_L_4,
    "SiT-L/8": SiT_L_8,
    "SiT-L/1": SiT_L_1,
    "SiT-B/1": SiT_B_1,
    "SiT-B/2": SiT_B_2,
    "SiT-B/4": SiT_B_4,
    "SiT-B/8": SiT_B_8,
    "SiT-S/2": SiT_S_2,
    "SiT-S/4": SiT_S_4,
    "SiT-S/8": SiT_S_8,
}


class CrossAttention(nn.Module):  # Modified from nano-GPT
    """Similar to CausalSelfAttention but without causal mask"""

    def __init__(self, n_head=12, n_embd=768, n_cond=1024, dropout=0.0, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_kv_attn = nn.Linear(n_cond, 2 * n_embd, bias=bias)
        self.c_q_attn = nn.Linear(n_embd, n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

    def forward(self, x, cond, attn_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        _, CT, _ = cond.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_q_attn(x)
        k, v = self.c_kv_attn(cond).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh,  T, hs)
        k = k.view(B, CT, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, CT, hs)
        v = v.view(B, CT, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, CT, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, CT) -> (B, nh, T, CT)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # mask out attention weights if requested
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # nan to num
        # y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CondDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0, n_cond=1024, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Fix: Set n_cond to hidden_size to match the projected condition
        self.cross_attn = CrossAttention(
            n_embd=hidden_size,
            n_cond=n_cond,  # Match DINO_v2 projected condition size
            # n_cond=hidden_size,  # Changed from default 512
            n_head=num_heads,
            dropout=0.0,
            bias=True,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        # Add projection from PointTransformer-v3 output (512) to hidden_size
        # self.cx_proj = nn.Linear(512, hidden_size)

    def forward(self, x, t, cx=None):
        """t is typically time embedding (or mixed with label embedding)
        cx is more general sequential conditional signals like image patch embeddings or text embeddings.
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        if cx is not None:
            # Project cx from (B, seq_len, 32) to (B, seq_len, hidden_size)
            # cx_projected = self.cx_proj(cx)
            x = x + gate_mca.unsqueeze(1) * self.cross_attn(
                # modulate(self.norm2(x), shift_msa, scale_msa), cx_projected
                modulate(self.norm2(x), shift_msa, scale_msa),
                cx,
            )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cond_dim=512,
        class_dropout_prob=0.1,
        learn_sigma=True,
        conditional=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.conditional = conditional

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.num_latents = 512  # Store this from pc_embedder config
        self.pc_embedder = PC_Encoder(depth=24, dim=cond_dim, num_inputs=16384, num_latents=self.num_latents)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                CondDiTBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, n_cond=cond_dim
                )
                for _ in range(depth)
            ]
        )
        self.no_cond_vector = nn.Parameter(torch.zeros(1, 1, cond_dim))
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, cond1=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        cond1: Point cloud condition (N, num_points, 3)
        """
        bs = x.shape[0]
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        pc_embed = self.pc_embedder(cond1)  # (N, num_latents, cond_dim)

        if self.training:
            drop_mask = torch.rand(bs, device=pc_embed.device) < 0.1
            null_expanded = self.no_cond_vector.expand(bs, self.num_latents, -1)
            pc_embed = torch.where(
                drop_mask[:, None, None],
                null_expanded,
                pc_embed,
            )

        return self._forward_with_embed(x, t, pc_embed)

    def _forward_with_embed(self, x, t, pc_embed):
        """Forward pass with already-embedded condition."""
        for block in self.blocks:
            x = block(x, cx=pc_embed, t=t)
        x = self.final_layer(x, c=t)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x


    def forward_with_cfg(self, x, t, cond1, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        cond1: Point cloud condition (N, num_points, 3)
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        t_combined = torch.cat([t[:len(t)//2], t[:len(t)//2]], dim=0)
        
        x_emb = self.x_embedder(combined) + self.pos_embed
        t_emb = self.t_embedder(t_combined)
        
        # Embed only the conditional point clouds
        bs_half = cond1.shape[0] // 2
        pc_embed_cond = self.pc_embedder(cond1[:bs_half])  # (N/2, num_latents, cond_dim)
        pc_embed_null = self.no_cond_vector.expand(bs_half, self.num_latents, -1)
        
        # Concatenate: first half conditioned, second half null
        pc_embed_combined = torch.cat([pc_embed_cond, pc_embed_null], dim=0)
        
        model_out = self._forward_with_embed(x_emb, t_emb, pc_embed_combined)
        
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    


class AutoEncoder(nn.Module):
    def __init__(self, scaling_factor, **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        self.cfg = OmegaConf.create(kwargs)

        self.model_id = self.cfg.model.network.model_id

        assert self.cfg.model.network.channel_output <= 4
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        self.vae.config.scaling_factor = scaling_factor
        assert self.cfg.model.network.channel_input == 3

        # Reinitialize weights if training from scratch
        if self.cfg.model.network.get("from_scratch", False):
            self._reinitialize_weights(self.vae)

        if self.cfg.model.network.channel_output != 3:
            new_conv = nn.Conv2d(
                128,
                self.cfg.model.network.channel_output,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            new_conv.weight.data[:3, :, :, :] = self.vae.decoder.conv_out.weight.data
            new_conv.bias.data[:3] = self.vae.decoder.conv_out.bias.data
            new_conv.weight.data[3:, :, :, :].zero_()
            new_conv.bias.data[3:].zero_()
            self.vae.decoder.conv_out = new_conv

        self.lr = self.cfg.model.optimizer.lr
        self.train_encoder = self.cfg.model.network.train_encoder

        self.trainable_params = []
        self.trainable_params += [(list(self.vae.decoder.parameters()), 1.0)]
        if self.train_encoder:
            self.trainable_params += [(list(self.vae.encoder.parameters()), 1.0)]

    def _reinitialize_weights(self, model):
        """Reinitialize all weights in the model for training from scratch."""
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @torch.no_grad()
    def encode_image(self, x_input):
        b = x_input.shape[0]
        z = self.vae.encode(x_input).latent_dist
        z = z.sample()
        z = z * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def decode_latent(self, latents):
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def autoencode(self, x_input):
        z = self.encode_image(x_input)
        x_output = self.decode_latent(z)
        return x_output, z

    def forward(self, x_input):
        return self.autoencode(x_input)
