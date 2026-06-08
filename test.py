import torch
import pickle

from models.custom_network import FashionNet

model = FashionNet()

with open(
    "saved_models/model.pkl",
    "rb"
) as f:

    weights = pickle.load(f)

model.load_state_dict(weights)

print(
    "Model Loaded Successfully"
)
