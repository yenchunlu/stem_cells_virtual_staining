import os
import numpy as np
import cv2
from glob import glob
import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
""" Create directory for saving """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_X = sorted(glob(os.path.join(path,"training","images","*.jpg")))
    train_y = sorted(glob(os.path.join(path,"training","masks","*.jpg")))
    
    test_X = sorted(glob(os.path.join(path,"test","images","*.jpg")))
    test_y = sorted(glob(os.path.join(path,"test","masks","*.jpg")))

    return (train_X, train_y), (test_X, test_y)

def augment_data(images, masks, save_path, augment = True):
    size = (512, 512)
    index = 1
    for idx, (X, y) in tqdm.tqdm(enumerate(zip(images, masks)), total = len(images)):
        """ Extract the name """
        name = X.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        X = cv2.imread(X, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p = 1.0)
            augmented = aug(image = X, mask = y)

            X1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p = 1.0)
            augmented = aug(image = X, mask = y)
            X2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(p = 1.0)
            augmented = aug(image = X, mask = y)
            X3 = augmented["image"]
            y3 = augmented["mask"]

            X = [X, X1, X2, X3]
            Y = [y, y1, y2, y3]

        else:
            X = [X]
            Y = [y]
            

        
        for i, m in zip(X,Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{index:03}.png"
            tmp_mask_name = f"{index:03}.png"

            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)
            
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1



if __name__ == "__main__":
    np.random.seed(42)
    """ Load the data """
    data_path = "/Users/lu992/Documents/Unet implementation/data/"
    (train_X, train_y), (test_X, test_y) = load_data(data_path)
    
    print(f"Train:\ninputs - {len(train_X)} labels - {len(train_y)}")
    print(f"Test:\ninputs - {len(test_X)} labels - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("new_data/train/images/")
    create_dir("new_data/train/masks/")
    create_dir("new_data/test/images/")
    create_dir("new_data/test/masks/")

    """ Data augmentation """
    augment_data(train_X, train_y,"new_data/train/", augment=True)
    augment_data(test_X, test_y,"new_data/test/", augment=False)