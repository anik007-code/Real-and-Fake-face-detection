import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from configs.config_ml_model import BATCH_SIZE, MODEL_PATH
from functions import save_classification


def get_classification(data_preprocess, data_len, dir_name):
   model = load_model(f"{dir_name}/{MODEL_PATH}/model.h5")
   test = data_preprocess["test_Gen"]
   test.reset()
   pred = model.predict(x=test, steps=(data_len["total_test"] // BATCH_SIZE) + 1)

   pred = np.argmax(pred, axis=1)

   report = classification_report(test.classes, pred,
      target_names=test.class_indices.keys())

   save_classification(report, dir_name)