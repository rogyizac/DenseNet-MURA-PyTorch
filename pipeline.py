import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

data_cat = ['train', 'valid'] # data categories

def get_study_level_data(studies):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes, 
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset 
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    
    for phase in data_cat:
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for study_type in studies:
            BASE_DIR = '../MURA-v1.1/%s/%s/' % (phase, study_type)
            patients = list(os.walk(BASE_DIR))[0][1] # list of patient folder names        
            for patient in tqdm(patients): # for each patient folder
                for study in os.listdir(BASE_DIR + patient): # for each study in that patient folder
                    label = study_label[study.split('_')[1]] # get label 0 or 1
                    path = BASE_DIR + patient + '/' + study + '/' # path to this study
                    file_names = [file for file in os.listdir(path) if file.startswith('image')]
                    for file in file_names:
                        study_data[phase].loc[i] = [path + file, len(file_names), label] # add new row
                        i+=1
    return study_data

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        image = pil_loader(study_path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.df.iloc[idx, 2]
        return image, label

def get_dataloaders(data, batch_size=8, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in data_cat}
    return dataloaders

if __name__=='__main__':
    pass
