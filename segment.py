import time
start_time = time.time()
from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image
import numpy as np

prompt = "woman"

model = LangSAM()
print(model)
image_pil = Image.open("./images/park.png").convert("RGB")

masks, boxes, phrases, logits = model.predict(image_pil, prompt)

masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
final_mask = np.sum(masks_np, axis=0)
final_mask[final_mask == 1] = 255

mask_image = Image.fromarray(final_mask).convert("RGB")
# mask_image.show()

# exit()
# final_mask = masks_np[0].copy()
# for mask in masks_np[1:]:
#     final_mask[mask == 255] = 255

# mask_image = Image.fromarray(masks_np[0]).convert("RGB")
# mask_image = Image.fromarray(final_mask).convert("RGB")
# mask_image.show()
mask_image.save("./images/mask.png")

print("--- %s seconds ---" % (time.time() - start_time))
# Approx time: 4.75 mins