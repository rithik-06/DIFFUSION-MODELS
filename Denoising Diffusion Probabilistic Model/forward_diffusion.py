import torch


# Noise schedule
T = 1000
beta = torch.linspace(0.0001, 0.02, T)

# alpha = 1 - beta
alpha = 1 - beta

# alpha_bar = cumulative product
alpha_bar = torch.cumprod(alpha, dim=0)


def forward_diffusion(x0, t):
    """
    x0 : original image tensor [B,C,H,W]
    t  : timestep tensor [B]

    returns:
        xt     -> noisy image
        noise  -> actual gaussian noise
    """

    # Gaussian noise
    noise = torch.randn_like(x0)

    # reshape for broadcasting
    a_bar = alpha_bar[t].view(-1, 1, 1, 1)

    # DDPM forward equation
    xt = (
        torch.sqrt(a_bar) * x0 +
        torch.sqrt(1 - a_bar) * noise
    )

    return xt, noise




# batch of images
x0 = torch.randn(8, 3, 64, 64)

# random timestep
t = torch.randint(0, T, (8,))

xt, noise = forward_diffusion(x0, t)

print(xt.shape)