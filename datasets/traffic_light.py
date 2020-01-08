from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import random


class TrafficLight(Dataset):
    def __init__(self, root_, train=True, transform=None, target_transform=None):
        samples_list = []
        self.transform = transform
        self.target_transform = target_transform
        self.trainable = train

        data_path = Path(root_)
        class_path = [
            data_path / 'red',
            data_path / 'green',
            data_path / 'yellow',
            data_path / 'black'
        ]

        for i, path in enumerate(class_path):
            for filename in path.iterdir():
                sample_ = (str(filename), i)
                samples_list.append(sample_)

        random.shuffle(samples_list)
        if self.trainable:
            self.samples_list = samples_list[:int(len(samples_list)*0.9)+1]
            print("There are {} samples in training set.".format(len(self.samples_list)))
        else:
            self.samples_list = samples_list[int(len(samples_list) * 0.9)+1:]
            # self.samples_list = list(filter(lambda x: x[1]==3, samples_list))  # select all green class.
            print("There are {} samples in testing set.".format(len(self.samples_list)))

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        sample_ = self.samples_list[idx]
        img_path, target = sample_[0], sample_[1]
        img = Image.open(img_path)
        original_size = img.size

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, original_size
