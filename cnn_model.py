from functions import make_version, make_dir
from configs.config_ml_model import ROOT_DIR, REPORT_PATH
from preprocess.preprocess import preprocess
from training.training import train
from validation.validate import get_classification

version = make_version(f"{ROOT_DIR}/{REPORT_PATH}")
dir_name = f"{ROOT_DIR}/{REPORT_PATH}/{version}"
make_dir(dir_name)

data_preprocess, data_len = preprocess()

train(data_preprocess, data_len, dir_name)
get_classification(data_preprocess, data_len, dir_name)
