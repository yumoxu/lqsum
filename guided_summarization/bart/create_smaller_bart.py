import torch
from pathlib import Path

MODEL_PATH = Path('/home/lqsum/model/bart.large/')
SIZE = 640

model = torch.load(MODEL_PATH/'model.pt')

print(model["model"]["encoder.embed_positions.weight"].size())
print(model["model"]["decoder.embed_positions.weight"].size())

model["model"]["encoder.embed_positions.weight"] = model["model"]["encoder.embed_positions.weight"][:SIZE+2]
model["model"]["decoder.embed_positions.weight"] = model["model"]["decoder.embed_positions.weight"][:SIZE+2]

torch.save(model, MODEL_PATH / f'model_{SIZE}.pt')
