import os
import numpy as np

def get_validation_images():
    X = []
    
    sorted_files = sorted(os.listdir(VALID_DIR))
    for filename in sorted_files:
        file_path = os.path.join(VALID_DIR, filename)
        img = cv2.imread(file_path)
        X.append(img)
    
    X_arr = np.asarray(X)
    return X_arr

def get_validation_masks():
    Y = []

    sorted_files = sorted(os.listdir(VALID_MASK_DIR))
    for filename in sorted_files:
        file_path = os.path.join(VALID_MASK_DIR, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        Y.append(img)
    
    Y_arr = np.asarray(Y)
    Y_arr = Y_arr.reshape(Y_arr.shape[0], Y_arr.shape[1], Y_arr.shape[2], 1)
    return Y_arr

def image_generator(dir_path, batch_size):
    while True:
        curr_batch_count = 0
        curr_batch_X = []
        
        sorted_files = sorted(os.listdir(dir_path))
        for filename in sorted_files:
            file_path = os.path.join(dir_path, filename)
            img = cv2.imread(file_path)
            curr_batch_X.append(img)
            
            curr_batch_count += 1
            if curr_batch_count == batch_size:
                curr_batch_X = np.asarray(curr_batch_X)
                yield curr_batch_X
                
                curr_batch_X = []
                curr_batch_count = 0
        
        if curr_batch_count > 0:
            yield curr_batch_X
            
def mask_generator(dir_path, batch_size):
    while True:
        curr_batch_count = 0
        curr_batch_Y = []
        
        sorted_files = sorted(os.listdir(dir_path))
        for filename in sorted_files:
            file_path = os.path.join(dir_path, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            curr_batch_Y.append(img)
            
            curr_batch_count += 1
            if curr_batch_count == batch_size:
                curr_batch_Y = np.asarray(curr_batch_Y)
                curr_batch_Y[curr_batch_Y > 0] = 1
                curr_batch_Y = curr_batch_Y.reshape(curr_batch_Y.shape[0], 
                    curr_batch_Y.shape[1], curr_batch_Y.shape[2], 1)
                yield curr_batch_Y
                
                curr_batch_Y = []
                curr_batch_count = 0
        
        if curr_batch_count > 0:
            yield curr_batch_Y
            
def train_generator(batch_size):
    image_gen = image_generator(TRAIN_DIR, batch_size)
    mask_gen = mask_generator(MASK_DIR, batch_size)
    
    while True:
        yield (next(image_gen), next(mask_gen))

def encode_prediction(img):
        flat_img = img.flatten()
        flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
        flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])

        starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
        ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
        starts_ix = np.where(starts)[0] + 1
        ends_ix = np.where(ends)[0] + 1
        lengths = ends_ix - starts_ix

        encoding = ''
        for idx in range(len(starts_ix)):
            encoding += '%d %d ' % (starts_ix[idx], lengths[idx])
        return encoding
    
def decode_prediction(mask_rle, shape=(1280,1918)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)