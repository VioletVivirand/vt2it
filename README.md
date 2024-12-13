- [Usage](#usage)
  - [RealESRGAN (Re-write from ai-forever/Real-ESRGAN)](#realesrgan-re-write-from-ai-foreverreal-esrgan)


# Usage

## RealESRGAN (Re-write from [ai-forever/Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN))

Download sample image:

```bash
curl https://raw.githubusercontent.com/ai-forever/Real-ESRGAN/refs/heads/main/inputs/lr_image.png -o lr_image.png
```

Initialize Real-ESRGAN:

```python
from PIL import Image
from vt2it.upscale.airforever import RealESRGAN

# Read sample image
image = Image.open("./lr_image.png") 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(scale=4,)  # scale Options: 2 for x2, 4 for x4, 8 for x8
model.to(device)

image_upscaled = model(image)
```