from PIL import Image
import os
import sys


def main(argv):
    dirName = "resize"
    os.makedirs(dirName, exist_ok=True)

    if len(argv) == 2:
        img_path = os.path.join(os.getcwd(), argv[1])
        lsdir = os.listdir(img_path)
        count = 0
        for img in lsdir:
            count += 1
            image = Image.open(os.path.join(img_path, img))
            new_image = image.resize((256, 256))
            new_image = new_image.convert('RGB')
            new_image.save(os.path.join(os.getcwd(), 'resize', f'{img}'))
            print(count, len(lsdir))


if __name__ == "__main__":
    main(sys.argv)
