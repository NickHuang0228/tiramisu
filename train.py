
from keras.models import Model
from keras.layers import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras_tiramisu.tiramisu.model import create_tiramisu
import glob
import numpy as np
from PIL import Image
masks = glob.glob("G:\\0609\\test\\*.png") ## 　修改路徑




orgs = list(map(lambda x: x.replace(".png", ".jpg"), masks))

imgs_list = []
masks_list = []

#影像resize
for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((480,480))))
    masks_list.append(np.array(Image.open(mask).resize((480,480))))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)
# 確定影像 shape
print(imgs_np.shape, masks_np.shape)


## Get data into correct shape, dtype and range (0.0-1.0)
print(imgs_np.max(), masks_np.max())

x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)/255
# 應該要都 1
print(x.max(), y.max())
##(x, 480, 480, 3) (y, 480, 480)
print(x.shape, y.shape)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 3)
##(x, 480, 480, 3) (y, 480, 480, 1)
print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)
# ## 資料增量　
from keras_unet.utils import get_augmented
#


train_gen = get_augmented(
    x_train, y_train, batch_size=1000,
    data_gen_args = dict(
        rotation_range=90.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    ))

def main(args=None):

    # parse arguments


    input_shape = (480, 480, 3)
    img_input = Input(shape=input_shape)
    x = create_tiramisu(2, img_input)
    model = Model(img_input, x)
    from keras.optimizers import Adam, SGD
    from keras_unet.metrics import iou, iou_thresholded
    from keras_unet.losses import jaccard_distance
    model.compile(
        optimizer=Adam(),
        # optimizer=SGD(lr=0.01, momentum=0.99),
        loss=jaccard_distance,
        # loss='binary_crossentropy',
        metrics=[iou, iou_thresholded]
    )
    checkpoint = ModelCheckpoint("G:\\test.h5", monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False,
                                 mode='max')

    callbacks_list = [checkpoint]
    model.fit_generator(train_gen, steps_per_epoch=1000,
                                  epochs=100000, callbacks=callbacks_list, verbose=1,
                                  validation_data=(x_val, y_val), shuffle=True)

    model.save_weights(args.output_path)


if __name__ == '__main__':
    main()




