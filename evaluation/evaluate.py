from sklearn.metrics import confusion_matrix
from sklearn import metrics
import os
import cv2
from tensorflow.keras.models import load_model
from configs.config_data import TEST_DATA_PATH, DATA_PATH
from configs.config_ml_model import ROOT_DIR, MODEL_VERSION_IN_WORK, MODEL_PATH, REPORT_PATH
from functions import img_process, create_montages, save_classification, draw_plot_confusion_matrix, get_latest_version


def evaluation(dir_name):
    model = load_model(f"{ROOT_DIR}/{REPORT_PATH}/{MODEL_VERSION_IN_WORK}/{MODEL_PATH}/model.h5")
    categories_folder = f"{ROOT_DIR}/{DATA_PATH}/{TEST_DATA_PATH}"

    category_true = []
    category_pred = []
    results = []

    categories = os.listdir(categories_folder)
    for category in categories:
        image_path = f"{categories_folder}/{category}"
        image_list = os.listdir(f"{categories_folder}/{category}")
        print(image_list)
        for image in image_list:
            org = cv2.imread(f"{image_path}/{image}")
            image = img_process(org)
            pred = model.predict(image)
            pred = pred.argmax(axis=1)[0]

            label = ['fake', 'real']

            org = cv2.resize(org, (128, 128))
            cv2.putText(org, label[pred], (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            results.append(org)

            category_true.append(categories.index(category))
            category_pred.append(pred)

    classification_accuracy = metrics.classification_report(category_pred, category_true)

    save_classification(classification_accuracy, dir_name)

    data = confusion_matrix(category_true, category_pred)
    draw_plot_confusion_matrix(data, categories, dir_name)

    create_montages(results, 10, 96, dir_name)


def evaluate_image(img_path):
    # file_path = f"{ROOT_DIR}/{REPORT_PATH}"
    # latest_version = get_latest_version(file_path)
    # model = load_model(f"{file_path}/{latest_version}/{MODEL_PATH}/model.h5")
    model = load_model('/home/anik/Projects/misc-ml/liveliness_detection/REPORT/1/MODEL/model.h5')

    org = cv2.imread(f"{ROOT_DIR}/{img_path}")
    image = img_process(org)
    pred = model.predict(image)
    pred = pred.argmax(axis=1)[0]
    if pred == 0:
        return "fake"
    elif pred == 1:
        return "real"
