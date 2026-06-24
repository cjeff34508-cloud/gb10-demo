from transformers import AutoTokenizer, AutoModel
import torch

# Load a small model that's already downloaded
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
model = model.to("cuda")

# Test inference
text = "A dog sitting on grass"
print(f"Model loaded: {model.config.model_type}")
print(f"GPU available: {torch.cuda.is_available()}")
