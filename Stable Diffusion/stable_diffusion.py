"""
Minimal Stable Diffusion (Latent Diffusion Model) in PyTorch.

Stable Diffusion = 3 pieces glued together:

    1. VAE        -> compresses image 64x64x3  into latent 8x8x4
    2. Text encoder -> turns a prompt into token embeddings [B, L, D]
    3. UNet       -> denoises the LATENT, conditioned on text via cross-attention

Diffusion happens in latent space (small + cheap), not pixel space.
Everything here is tiny so it runs on CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. VAE  (image <-> latent)
# ----------------------------------------------------------------------

class VAE(nn.Module):
    """Downsamples 64x64 -> 8x8 (factor 8, same as real SD)."""

    def __init__(self, latent_ch=4, base=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1),      # 64 -> 32
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),  # 32 -> 16
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),  # 16 -> 8
            nn.SiLU(),
            # 2 * latent_ch  ->  mean and logvar
            nn.Conv2d(base * 4, latent_ch * 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, base * 4, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),                     # 8 -> 16
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),                     # 16 -> 32
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),                     # 32 -> 64
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)     # reparameterization trick
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)


# ----------------------------------------------------------------------
# 2. Text encoder  (stand-in for CLIP)
# ----------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Token ids -> contextual embeddings. Real SD uses a frozen CLIP here."""

    def __init__(self, vocab_size=1000, dim=128, max_len=16, layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))

        block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)

    def forward(self, tokens):
        h = self.token_emb(tokens) + self.pos_emb[:, : tokens.shape[1]]
        return self.transformer(h)                 # [B, L, dim]


# ----------------------------------------------------------------------
# 3. UNet building blocks
# ----------------------------------------------------------------------

def timestep_embedding(t, dim):
    """Sinusoidal embedding of the timestep, same idea as positional encoding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Conv block that also receives the timestep embedding."""

    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.t_proj = nn.Linear(t_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]   # inject time
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Query = image latent, Key/Value = text. This is how the prompt steers generation."""

    def __init__(self, ch, ctx_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)

        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(ctx_dim, ch)
        self.to_v = nn.Linear(ctx_dim, ch)
        self.out = nn.Linear(ch, ch)

    def forward(self, x, context):
        B, C, H, W = x.shape

        h = self.norm(x).view(B, C, H * W).transpose(1, 2)     # [B, HW, C]

        q = self.to_q(h)
        k = self.to_k(context)
        v = self.to_v(context)

        # split into heads -> [B, heads, seq, head_dim]
        def split(t):
            return t.view(B, -1, self.heads, C // self.heads).transpose(1, 2)

        a = F.scaled_dot_product_attention(split(q), split(k), split(v))
        a = a.transpose(1, 2).reshape(B, H * W, C)

        a = self.out(a).transpose(1, 2).view(B, C, H, W)
        return x + a                                            # residual


class UNet(nn.Module):
    """Predicts the noise added to a latent, given timestep + text context."""

    def __init__(self, latent_ch=4, base=64, ctx_dim=128, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        self.in_conv = nn.Conv2d(latent_ch, base, 3, padding=1)

        # down: 8x8 -> 4x4
        self.down1 = ResBlock(base, base, t_dim)
        self.attn1 = CrossAttention(base, ctx_dim)
        self.downsample = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)

        # middle: 4x4
        self.mid1 = ResBlock(base * 2, base * 2, t_dim)
        self.mid_attn = CrossAttention(base * 2, ctx_dim)
        self.mid2 = ResBlock(base * 2, base * 2, t_dim)

        # up: 4x4 -> 8x8
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv = nn.Conv2d(base * 2, base, 3, padding=1)
        self.up1 = ResBlock(base * 2, base, t_dim)   # *2 because of skip connection
        self.attn2 = CrossAttention(base, ctx_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, latent_ch, 3, padding=1)
        )

    def forward(self, z, t, context):
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))

        h = self.in_conv(z)
        h = self.attn1(self.down1(h, t_emb), context)
        skip = h

        h = self.downsample(h)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid2(h, t_emb)

        h = self.up_conv(self.upsample(h))
        h = self.up1(torch.cat([h, skip], dim=1), t_emb)
        h = self.attn2(h, context)

        return self.out(h)


# ----------------------------------------------------------------------
# 4. Diffusion schedule (DDPM)
# ----------------------------------------------------------------------

class Diffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, z0, t, noise):
        """q(z_t | z_0) -- the forward process, done in one shot."""
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return a_bar.sqrt() * z0 + (1 - a_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, unet, context, uncond_context, shape, guidance=7.5, device="cpu"):
        """Reverse process with classifier-free guidance."""
        z = torch.randn(shape, device=device)

        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)

            # two passes: with prompt and without
            eps_cond = unet(z, t, context)
            eps_uncond = unet(z, t, uncond_context)

            # push the prediction away from "unconditional" -> stronger prompt effect
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)

            a = self.alpha[i]
            a_bar = self.alpha_bar[i]

            # mean of p(z_{t-1} | z_t)
            z = (z - ((1 - a) / (1 - a_bar).sqrt()) * eps) / a.sqrt()

            if i > 0:
                z = z + self.beta[i].sqrt() * torch.randn_like(z)

        return z


# ----------------------------------------------------------------------
# 5. Full model
# ----------------------------------------------------------------------

class StableDiffusion(nn.Module):
    def __init__(self, latent_ch=4, ctx_dim=128, vocab_size=1000):
        super().__init__()
        self.vae = VAE(latent_ch=latent_ch)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, dim=ctx_dim)
        self.unet = UNet(latent_ch=latent_ch, ctx_dim=ctx_dim)

        # real SD scales latents so they have ~unit variance
        self.latent_scale = 0.18215

    def loss(self, images, tokens, diffusion):
        """Training objective: predict the noise that was added to the latent."""
        z, _, _ = self.vae.encode(images)
        z = z * self.latent_scale

        context = self.text_encoder(tokens)

        t = torch.randint(0, diffusion.T, (z.shape[0],), device=z.device)
        noise = torch.randn_like(z)
        z_t = diffusion.add_noise(z, t, noise)

        pred = self.unet(z_t, t, context)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def generate(self, tokens, diffusion, guidance=7.5):
        B = tokens.shape[0]
        device = tokens.device

        context = self.text_encoder(tokens)
        # empty prompt (token 0) = unconditional branch
        uncond = self.text_encoder(torch.zeros_like(tokens))

        z = diffusion.sample(
            self.unet, context, uncond,
            shape=(B, 4, 8, 8), guidance=guidance, device=device,
        )
        return self.vae.decode(z / self.latent_scale)


# ----------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StableDiffusion().to(device)
    diffusion = Diffusion(T=1000, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # fake data: 4 images + 4 prompts of 16 tokens
    images = torch.randn(4, 3, 64, 64, device=device).clamp(-1, 1)
    tokens = torch.randint(1, 1000, (4, 16), device=device)

    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    for step in range(5):
        loss = model.loss(images, tokens, diffusion)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step}  loss {loss.item():.4f}")

    # sampling with the full 1000 steps is slow -> use a short schedule for the demo
    demo_diffusion = Diffusion(T=50, device=device)
    imgs = model.generate(tokens[:2], demo_diffusion)
    print("generated:", imgs.shape)
