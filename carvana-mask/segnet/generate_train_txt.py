import os
import sys

def generate_train_txt(images_dir, masks_dir):
    with open(os.path.join(images_dir, '../train.txt'), 'w') as output:
        for file in os.listdir(images_dir):
            image_path = os.path.abspath(os.path.join(images_dir, file))
            mask_path = os.path.abspath(os.path.join(masks_dir, file))

            output.write('%s %s\n' % (image_path, mask_path))

if __name__ == '__main__':
    generate_train_txt(sys.argv[1], sys.argv[2])

