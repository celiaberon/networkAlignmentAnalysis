import json
import os

from PIL import Image
from torch.utils.data import Dataset


def organize_imagenet_tiny(data_dir):

    '''
    Ensure data is formatted as dataset/label/img.png to load with ImageFolder.
    If data doesn't exist, can be downloaded with:
        wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    '''

    # Create separate subfolders for images based on their labels indicated
    # in the val_annotations txt file (validation only for imagenet-tiny).
    img_dir = os.path.join(data_dir, 'images')

    # Store img filename (word 0) and label (word 1) for every line in the txt
    # file (as key value pair).
    img_dict = {}

    # Open and read val annotations text file
    with open(os.path.join(data_dir, 'val_annotations.txt'), 'r') as f:
        data = f.readlines()

        for line in data:
            words = line.split('\t')  # split words using tabs
            img_dict[words[0]] = words[1]

    # Create category (label) subfolders if necessary and reset image paths.
    for img, folder in img_dict.items():
        newpath = os.path.join(img_dir, folder)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


class ImageNetKaggle(Dataset):

    '''
    Custom class to load ImageNet dataset as Dataset module in PyTorch.
    Citation: https://github.com/paulgavrikov/pytorch-image-models/blob/main/timm/data/dataset_factory.py
    Requires previous execution of:
    wget https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json
    wget https://gist.githubusercontent.com/paulgavrikov/3af1efe6f3dff63f47d48b91bb1bca6b/raw/00bad6903b5e4f84c7796b982b72e2e617e5fde1/ILSVRC2012_val_labels.json
    in ImageNet directory.
    '''

    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]
