import time
start_time = time.time()
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting


models = [
    "runwayml/stable-diffusion-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
]

model = models[1]

prompt = "puppies sitting on a park bench, high resolution"

pipe = AutoPipelineForInpainting.from_pretrained(
    model,
    # variant="fp16", # uncomment if using stable diffusion variant
    torch_dtype=torch.float32,
    use_safetensors=True,
    safety_checker = None
)

pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing("max")

image_pil = Image.open("./images/park.png")
mask_image = Image.open("./images/mask.png")

image = pipe(prompt=prompt, image=image_pil, mask_image= mask_image).images[0]
image.save("./images/output.png")

print("--- %s seconds ---" % (time.time() - start_time))
# Approx time: 2.4 mins