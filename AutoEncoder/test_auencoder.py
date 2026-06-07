from autoencoder_from_scratch import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def test_autoencoder():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.view(-1, 784).to(device)
            z = model.encoder(x)
            latents.append(z.cpu())
            labels.append(y)

    latents = torch.cat(latents)
    labels = torch.cat(labels)

    assert latents.shape[0] == labels.shape[0], "Number of latents and labels should match"
    assert latents.shape[1] == 32, "Latent dimension should be 32"
    print("Test passed: Latent representations and labels are correctly generated.")