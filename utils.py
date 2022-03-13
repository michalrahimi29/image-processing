from PIL import Image
import torchvision
from pathlib import Path
import pandas as pd
import os
import torchvision.transforms as transforms
from math import floor
import matplotlib.pyplot as plt


# return list of (img,label)
def readImages(dir, imageSize, classification=False):
    # process one given raw image
    def processImg(rawImg):
        image = Image.open(rawImg)
        image = transforms.CenterCrop((image.height * 0.8, image.width * 0.8))(image)
        image = image.resize((imageSize[0], imageSize[1]))
        image = torchvision.transforms.ToTensor()(image)
        #image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return image

    # read ages from excel file
    fileName = f"boneage-{dir}.csv"
    # if dir == "train":
    #     dir += "_small"

    id2age = {}
    csvFile = pd.read_csv(f"images\\{dir}\\{fileName}")
    ages = csvFile.pop("boneage").tolist()

    # for multiclass classification 19 classes
    if classification:
        ages = [floor(age/12.0) for age in ages]



    ids = csvFile.pop("id").tolist()

    # build dict from image id to age
    for i in range(len(ages)):
        id2age[ids[i]] = ages[i]

    path = Path(f"images\\{dir}")
    # Store the image file names in a list as long as they are jpgs
    ids = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.png']

    # read images, attach image i to age i
    images = []
    for i, rawImg in enumerate(Path(f"images\\{dir}").glob('*.png')):  # subDir is long number
        name = int(ids[i].split(".")[0])
        images.append((processImg(rawImg), id2age[name]))
    return images




