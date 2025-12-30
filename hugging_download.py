import os
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
import argparse


# --- Parse Arguments ---
parser = argparse.ArgumentParser()

parser.add_argument("--img_size")
parser.add_argument("--column")
parser.add_argument("--link")
parser.add_argument("--split")

args = parser.parse_args()

img_size = args.img_size

# --- Dataset Loading ---
link = args.link
split = args.split
column = args.column


class DatasetTransform(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        item = self.dataset[index][column]
        if self.transform:
            item = self.transform(item)
        return item

def convert_to_rgb(x):
    return x.convert("RGB")


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

ds = load_dataset(link)
train = ds[split]
dataset = DatasetTransform(train, transform)

if not os.path.exists(link):
    os.makedirs(link)

for i in range(len(dataset)):
    img = train[i][column]
    img.save(link + f"/image{i}.png")

