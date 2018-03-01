from my_constants import *
from data_utils import decode_prediction
from matplotlib import pyplot as plt
import pandas as pd
import make_single_blob
import sys

def view_mask(mask, img_id='Mask'):
    plt.title(img_id)
    plt.imshow(mask)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv(OUTPUT_FILE, index_col=0)
    if len(sys.argv) > 1:
        img_id = sys.argv[1]
        mask = decode_prediction(df.get_value(img_id, 'rle_mask'), (1280,1918))
        view_mask(mask, img_id)
    else:
        img_ids = df.index
        for  img_id in reversed(img_ids):
            mask = decode_prediction(df.get_value(img_id, 'rle_mask'), (1280,1918))
            view_mask(mask, img_id)