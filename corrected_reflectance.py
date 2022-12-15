import csv
import random
from PIL import Image as plimg
import numpy as np
import torch as tr
from torchvision import transforms
import pickle as pkl

# use a seed to keep the training and testing datasets the same for all models we test
random_seed = 44

# convert from strings to int representation of the labels
def encode(label):
    if label in ['clear', 'water']:
        return 0
    else: # cloudy, land
        return 1

def create_correlation_matrix(data):
    imgs, weather, terrain = data
    num_examples = len(imgs)
    weather_count = 0
    terrain_count = 0
    clear_water_count = 0
    clear_land_count = 0
    cloud_water_count = 0
    cloud_land_count = 0
    for i in range(num_examples):
        encoded_weather = encode(weather[i])
        encoded_terrain = encode(terrain[i])
        weather_count += encoded_weather
        terrain_count += encoded_terrain
        if encoded_weather == 0:
            if encoded_terrain == 0:
                clear_water_count += 1
            else:
                clear_land_count += 1
        else:
            if encoded_terrain == 0:
                cloud_water_count += 1
            else:
                cloud_land_count += 1
                #      clear, cloudy, water, land
                #clear
                #cloudy
                #water
                #land
    result = {'nums': np.array([num_examples - weather_count, weather_count, num_examples - terrain_count, terrain_count]),
                'adj':[[0,0,clear_water_count, clear_land_count],
                        [0,0,cloud_water_count, cloud_land_count],
                        [clear_water_count, clear_land_count,0,0],
                        [cloud_water_count, cloud_land_count,0,0]]}
    with open('corrected_reflectance.pkl', 'wb') as handle:
        pkl.dump(result, handle, protocol=pkl.HIGHEST_PROTOCOL)

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
    create_correlation_matrix(training_data)
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
    