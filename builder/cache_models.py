
import torch
from diffusers import DiffusionPipeline

def cacheFluxSchnell():
  print("Fetch flux schnell from hugging face model hub")

  pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16, use_safetensors=True, add_watermarker=False)

  return pipe

if __name__ == "__main__":
  cacheFluxSchnell()