#!/usr/bin/env python
import click 
import os 
from os.path import join 
from PIL import Image 


@click.command()
@click.argument("source")
@click.argument("destination")
@click.option("--width", "-w", "width", default=64, type=int)
@click.option("--height", "-h", "height", default=64, type=int)
def main(source, destination, width, height):
    """
    Creates downscaled versions of an image dataset
    """
    print(source)
    os.makedirs(destination, exist_ok=True)

    classes = os.listdir(source) 
    all_images = []
    
    with click.progressbar(classes, label="Finding Images") as bar:
        for clazz in bar:
            path = join(source, clazz)
            files = os.listdir(path)
            all_images.extend([(clazz, f) for f in files])
    
    print(f"Processing {len(all_images)}")
    with click.progressbar(all_images, label="Processing") as bar:
        for clazz, f in bar:
            try:
                src = join(source, clazz, f)
                dst = join(destination, clazz, f)
                os.makedirs(join(destination, clazz), exist_ok=True)
                img = Image.open(src)
                img = img.resize((height,width), Image.ANTIALIAS)
                img.save(dst, "JPEG")
            except OSError as e:
                print(f"Error: {e} ({src})")
            except Exception:
                print(f"Unexpected Error: {e} ({src})")
             

if __name__ == "__main__":
    main()


