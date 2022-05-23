from keras import Sequential
from keras.applications.densenet import DenseNet201, layers
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D, \
    Activation, UpSampling2D, MaxPool2D, AveragePooling2D
import tensorflow as tf
from keras.models import Model

from configs.config_ml_model import EPOCHS, BATCH_SIZE, OPT
from functions import save_model_summary, save_model, save_accuracy, save_plot


def train(data_preprocess, data_len, dir_name):
    incepnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
    for layer in incepnet.layers:
        layer.trainable = False

    x = incepnet.output
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=incepnet.input, outputs=predictions)

    # model = Sequential([
    #     incepnet,
    #     GlobalAveragePooling2D(),
    #     Dense(300, activation=tf.nn.leaky_relu),
    #     BatchNormalization(),
    #     Dropout(0.5),
    #     Dense(2, activation=tf.nn.softmax)
    # ])

    # vgg = MobileNetV2(input_shape=(96, 96, 3), weights='imagenet', include_top=False)
    # for layer in vgg.layers:
    #     layer.trainable = False
    #
    # x = layers.Flatten()(vgg.output)
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # prediction = Dense(2, activation='softmax')(x)
    # model = Model(inputs=vgg.input, outputs=prediction)
    #
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
    #                            input_shape=(48, 48, 1)),
    #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     # tf.keras.layers.BatchNormalization(),
    #     # tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    #     tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256, activation=tf.nn.relu),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    #
    # ])

    # model = Sequential()
    # # first CONV => RELU => CONV => RELU => POOL layer set
    # model.add(Conv2D(16, (3, 3), padding="same",
    #                  input_shape=(48, 48, 1)))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Conv2D(32, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # # second CONV => RELU => CONV => RELU => POOL layer set
    # model.add(Conv2D(64, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Conv2D(128, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # # first (and only) set of FC => RELU layers
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    # # softmax classifier
    # model.add(Dense(128))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    #
    # model.add(Dense(2))
    # model.add(Activation("softmax"))

    # model = Sequential()
    #
    # model.add(Conv2D(32, (3, 3), padding="same", input_shape=(96, 96, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(Flatten())
    # model.add(Dense(2))
    # model.add(Activation("softmax"))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    summary = model.summary()
    save_model_summary(model, dir_name)

    histry = model.fit(
        x=data_preprocess["train_Gen"],
        steps_per_epoch=data_len["total_train"] // BATCH_SIZE,
        validation_data=data_preprocess["validation_Gen"],
        validation_steps=data_len["total_val"] // BATCH_SIZE,
        epochs=EPOCHS)

    save_model(model, dir_name)
    save_accuracy(histry, dir_name)
    save_plot(EPOCHS, histry, dir_name)
