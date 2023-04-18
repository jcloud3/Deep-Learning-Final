import torch
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

notes = """
https://medium.com/howtoai/pytorch-torchvision-coco-dataset-b7f5e8cad82
pytorch does not automatically download the dataset so we have to 
download it manually. In assignment 2, torchvion.datasets.CIFAR10 had
a download option

Installing pycocotools on Windows: pip install Cython && pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio": https://visualstudio.microsoft.com/downloads

813M    annotations
257M    annotations_trainval2017.zip
4.0K    get_data_coco.sh
 18G    train2017
 18G    train2017.zip
787M    val2017
785M    val2017.zip
"""

# Get paths
train_data_path = "./data/train2017"
train_json_path = "./data/annotations/captions_train2017.json"

validation_data_path = "./data/val2017"
validation_json_path = "./data/annotations/captions_val2017.json"

# Load training data
training_data = torchvision.datasets.CocoCaptions(
    root=train_data_path,
    annFile=train_json_path,
    transform=ToTensor(),
)

print(f"training count: {len(training_data)}")

img, target = training_data[0]

print(target)

img = np.array(img).transpose(1, 2, 0)
plt.imshow(img)
plt.show()
input("Press X to continue")
