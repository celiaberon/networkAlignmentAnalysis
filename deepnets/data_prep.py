import os


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