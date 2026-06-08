import torch
import pickle

from PIL import Image

from torchvision import transforms

from models.custom_network import FashionNet

model = FashionNet()

with open(
    "saved_models/model.pkl",
    "rb"
) as f:

    weights = pickle.load(f)

model.load_state_dict(weights)

model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

image = Image.open(
    "sample.png"
)

image = transform(image)

image = image.unsqueeze(0)

with torch.no_grad():

    output = model(image)

    pred = torch.argmax(
        output,
        dim=1
    )

print(
    "Predicted Class:",
    pred.item()
)