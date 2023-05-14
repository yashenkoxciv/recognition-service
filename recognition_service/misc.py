import io
import PIL
import urllib.request
from PIL import Image


def load_image(url: str) -> PIL.Image:
    response = urllib.request.urlopen(url)
    image_bytes = response.read()

    image = Image.open(io.BytesIO(image_bytes))
    return image

