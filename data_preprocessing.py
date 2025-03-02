import os
import cv2
import numpy as np


labels = ['PNEUMONIA','NORMAL']
img_size = 128
def get_data(data_dir):
    data=[]
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append((resized_arr, class_num))
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)

if __name__ == "__main__":
    train_data = get_data("chest_xray/train")
    print("Data preprocessing complete.")
