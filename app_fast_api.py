import os
# import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File
import shutil
from configs.config_ml_model import ROOT_DIR
from evaluation.evaluate import evaluate_image
from functions import make_dir_if_not_exists, remove_file_if_exists

app = FastAPI()

IMAGE_FOLDER = 'STATIC/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get('/')
async def home():
    return {"message": "Human Face Liveness Detection is working, Don't Worry!"}


@app.post('/profile_picture')
async def image(file: UploadFile = File(...)):
    make_dir_if_not_exists(IMAGE_FOLDER)
    # return{"filename" : file.filename}
    if file and allowed_file(file.filename):
        path = os.path.join(IMAGE_FOLDER, file.filename)
        with open(f"{ROOT_DIR}/{IMAGE_FOLDER}/{file.filename}", 'wb') as buffer:
            # print(buffer)
            shutil.copyfileobj(file.file, buffer)
        image_classifier = evaluate_image(path)
        remove_file_if_exists(path)
        return image_classifier
    else:
        return "File Format is not Compatible"


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')

# for running             uvicorn app_fast_api:app --reload
