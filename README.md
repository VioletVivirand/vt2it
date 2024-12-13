# vt2it: Vivi's GenAI Text-to-image (t2i) Toolbox <!-- omit in toc -->

For every developer who unintentionally visits here: Welcome my friend! This is my repo for storing things that will be use in GenAI Text-to-image projects for my own convenience.

Take anything you want without hesitation! But I am sorry and afraid that I won't receive any feature request. Just because this is my little playground, so I am not responsive for everything but my own needs.

But if something placed here violates your copyrights, **PLEASE CONTACT ME ASAP**.

Thank you so much! Have a nice day!

- [Usage](#usage)
  - [Real-ESRGAN](#real-esrgan)
    - [Rewrite from ai-forever/Real-ESRGAN](#rewrite-from-ai-foreverreal-esrgan)


# Usage

## Real-ESRGAN

Real-ESRGAN is an algorithm aims to generate image restoration. In GenAI projects, it generally adapted to upscale the images.

### Rewrite from [ai-forever/Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN)

Download sample image:

```bash
curl https://raw.githubusercontent.com/ai-forever/Real-ESRGAN/refs/heads/main/inputs/lr_image.png -o lr_image.png
```

To use Real-ESRGAN:

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