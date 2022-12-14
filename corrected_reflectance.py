import csv
import random
from PIL import Image as plimg
import numpy as np
import torch as tr
from torchvision import transforms

# use a seed to keep the training and testing datasets the same for all models we test
random_seed = 44

# convert from strings to int representation of the labels
def encode(label):
    if label in ['clear', 'water']:
        return 0
    else: # cloudy, land
        return 1

# Used for obtaining the training/testing data
def load_data(filename):
    imgs = []
    weather = []
    terrain = []
    with open(filename) as datacsv:
        reader = csv.DictReader(datacsv)
        for row in reader:
            imgs.append(row["filepath"])
            weather.append(row["weather"])
            terrain.append(row["terrain"])
    shufflelist = list(zip(imgs, weather, terrain))
    random.Random(random_seed).shuffle(shufflelist)
    imgs, weather, terrain = zip(*shufflelist)
    imgs, weather, terrain = list(imgs), list(weather), list(terrain)
    # split into training and test data (use 60% for training, 40% for testing)
    split_size = int(0.6 * len(imgs))
    training_data = (imgs[:split_size], weather[:split_size], terrain[:split_size])
    testing_data = (imgs[split_size:], weather[split_size:], terrain[split_size:])
    return training_data, testing_data

class CorrectedReflectanceDataset(tr.utils.data.Dataset):
    def __init__(self, data):
        self.imgs, self.weather, self.terrain = data
    
    def __getitem__(self, idx):
        # take the data sample by its index
        img = plimg.open(self.imgs[idx])
        # ditch the transparency
        img = img.convert('RGB')

        # Normalize the image and convert to tensor
        # First calculate the mean and standard deviation of pixel values
        npimg = np.array(img)
        mean = np.mean(npimg, axis=(0,1))
        std = np.std(npimg, axis=(0,1))
        transform2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        img = transform2(img)
        """transform2 = transforms.Compose([transforms.ToTensor()])
        img = transform2(img)"""
        # return the image and the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'weather': encode(self.weather[idx]),
                'terrain': encode(self.terrain[idx])
            }
        }
        return dict_data
    
    def __len__(self):
        return len(self.imgs)
    