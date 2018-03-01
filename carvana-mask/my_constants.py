import os

BASE_DIR = os.path.expanduser('~/Downloads/data/carvana-segmentation')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'validation')
VALID_MASK_DIR = os.path.join(BASE_DIR, 'validation_masks_jpg')
MASK_DIR = os.path.join(BASE_DIR, 'train_masks_jpg')
TEST_DIR = os.path.join(BASE_DIR, 'test')
OUTPUT_FILE = os.path.join(BASE_DIR, 'predictions.csv')
MODEL_SAVE = os.path.join(BASE_DIR, 'my_model.h5')

BATCH_SIZE_TRAIN = 2
BATCH_SIZE_INFER = 2