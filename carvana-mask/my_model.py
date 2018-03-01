from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from tqdm import tqdm_notebook as tqdm
from my_constants import *
import os
import math
import cv2

def get_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(1280,1918,3))
    x = base_model.output
    x = Conv2D(filters=4096, kernel_size=(7,7), name='fc1')(x)
    x = Conv2D(filters=1024, kernel_size=(1,1), name='fc2')(x)
    x = Conv2D(filters=1, activation='sigmoid', kernel_size=(1,1), name='prediction')(x)
    predictions = Conv2DTranspose(filters=1, kernel_size=(3,3), strides=(32,32), activation='linear')(x)
    model = Model(inputs=[base_model.input], outputs=[predictions])

    # make sure base layers are not trainable
    for layer in base_model.layers:
        layer.trainable = False
        
    # compile model
    metrics = ['accuracy']
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=metrics)

    return model

def train_model(model):
    model_checkpoint = ModelCheckpoint(MODEL_SAVE)
    early_stopping = EarlyStopping(verbose=1)
    train_gen = train_generator(BATCH_SIZE)
    num_steps = math.ceil(len(os.listdir(TRAIN_DIR))/BATCH_SIZE)
    model.fit_generator(train_gen, steps_per_epoch=num_steps, epochs=100, validation_data=(X_val, Y_val),
        callbacks=[early_stopping, model_checkpoint])

def make_predictions(model):
    # make predictions
    test_gen = image_generator(TEST_DIR, BATCH_SIZE_INFER)
    num_steps = math.ceil(len(os.listdir(TEST_DIR))/BATCH_SIZE_INFER)

    # because of memory constraints, predict each batch manually and then save results
    sorted_files = sorted(os.listdir(TEST_DIR))
    with open(OUTPUT_FILE, 'w') as output_file, \
        tqdm(total=num_steps*3) as pbar:
            output_file.write('img,rle_mask\n')
            for batch_num in range(num_steps):
                # get batch
                batch = next(test_gen)
                pbar.update(1)
                
                # predict on batch
                predictions = model.predict_on_batch(batch)
                pbar.update(1)

                # write output for batch
                for i, prediction in enumerate(predictions):
                    idx = batch_num*BATCH_SIZE + i
                    rle = encode_prediction(prediction)
                    output_file.write('%s,%s\n' % (sorted_files[idx], rle))
                pbar.update(1)