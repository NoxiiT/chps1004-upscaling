import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models import GeneratorResNet

# Utilise GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load("saved_models/generator_final.pth")
print(next(iter(state_dict.items())))

# === 1. Charger le générateur ===
generator = GeneratorResNet().to(device)
generator.load_state_dict(torch.load("saved_models/generator_final.pth", map_location=device))
generator.eval()

# === 2. Charger une image LR ===
img_lr = Image.open("test/lr_image.png").convert("RGB")

transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1,1]
            ])

lr_tensor = transform(img_lr).unsqueeze(0).to(device)

# === 3. Inférence ===
with torch.no_grad():
        sr_tensor = generator(lr_tensor)

        # === 4. Post-traitement et sauvegarde ===
        # Dé-normaliser (de [-1,1] à [0,1]) si Tanh a été utilisé
        sr_tensor = 0.5 * (sr_tensor + 1.0)
        sr_tensor = sr_tensor.clamp(0, 1)

        save_image(sr_tensor, "sr_output.png")

