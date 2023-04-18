"""
resize.py

The purpose of this preprocessing step is to make sure all the training images
are the same size so its easier for the training process.
"""

import os
from PIL import Image
import tqdm
import click

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    #for i, image in enumerate(images):
    for image in tqdm.tqdm(images, desc="Resizing Images"):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)


@click.command()
@click.option("--image_dir", type=str, default="image_captioning/data/train2014/")
@click.option("--output_dir", type=str, default="image_captioning/data/resized2014/")
@click.option("--image_size", type=int, default=256)
def cli(image_dir, output_dir, image_size):
    image_size = [image_size, image_size]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    cli()
