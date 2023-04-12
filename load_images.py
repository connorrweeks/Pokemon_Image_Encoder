import torchvision
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 

files = os.listdir("images/pokemon")

import pandas as pd
data = pd.read_csv("attributes.csv")
print(data.head())

ids = list()

type_1 = list(data["Type 1"])
type_2 = list(data["Type 2"])
df_names = list(data["Name"])

found_idx = [i for i, id in enumerate(data["#"]) if os.path.exists(f'./images/pokemon/{id}.png')]

all_images = []
for i in found_idx:
    file_name = data["#"][i] + ".png"
    tsr_img = torchvision.io.read_image('./images/pokemon/' + file_name)
    all_images.append(tsr_img[:3,:,:])
id = [data["#"][i] for i in found_idx]
names = [data["Name"][i] for i in found_idx]
id2name = {data["#"][i]:data["Name"][i] for i in found_idx}
id2type = {data["#"][i]:data["Type 1"][i] for i in found_idx}
name2type = {data["Name"][i]:data["Type 1"][i] for i in found_idx}
name2id = {data["Name"][i]:data["#"][i] for i in found_idx}

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data["Type 1"])

labels = []
for i, idx in enumerate(found_idx):
    
    label = torch.zeros(len(le.classes_))
    label[le.transform([data["Type 1"][idx]])[0]] = 1
    if(type(data["Type 2"][idx]) == str):
        label[le.transform([data["Type 2"][idx]])[0]] = 1
    labels.append(label)
#label_ids = le.transform(labels)

print("Number of types:", len(le.classes_))
print(le.classes_)

import random as r
r.seed(0)
zipped = list(zip(labels, all_images, names))
r.shuffle(zipped)
train_num = int(0.8 * len(zipped))
test_num = int(0.9 * len(zipped))
def to_samples(zipped):
    set_ids, set_images, set_names = zip(*zipped)
    return list(set_ids), list(set_images), list(set_names)
train_ids, train_images, train_names = to_samples(zipped[:train_num])
test_ids, test_images, test_names = to_samples(zipped[train_num:test_num])
val_ids, val_images, val_names = to_samples(zipped[test_num:])

#Augment training data images
import torchvision.transforms as T
base_transform = T.ToTensor()
aug_train_ids = train_ids[:]
aug_train_images = train_images[:]
aug_train_names = train_names[:]
policies = [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]
augmenters = [T.AutoAugment(policy) for policy in policies]
print("Augmenting Images...")
aug = augmenters[1]
for i, img in enumerate(train_images):
    #continue
    #for aug in augmenters:
    for _ in range(5):
        tensor_image = aug(img)
        aug_train_images.append(tensor_image)
        aug_train_ids.append(train_ids[i])
        aug_train_names.append(train_names[i])
train_ids = aug_train_ids
train_images = aug_train_images
train_names = aug_train_names

from torch.utils.data import Dataset, DataLoader
class CustomImageDataset(Dataset):
    def __init__(self, labels, images, names):
        self.labels = labels
        self.images = images
        self.names = names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = base_transform(np.transpose(self.images[idx].numpy(), (1, 2, 0)))
        return self.labels[idx], img, self.names[idx]


train_set = CustomImageDataset(train_ids, train_images, train_names)
test_set = CustomImageDataset(test_ids, test_images, test_names)
val_set = CustomImageDataset(val_ids, val_images, val_names)
all_set = CustomImageDataset(train_ids+test_ids+val_ids, train_images+test_images+val_images, train_names+test_names+val_names)

print(f"train:{len(train_set)} val:{len(val_set)} test:{len(test_set)}")

train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
all_loader = DataLoader(all_set, batch_size=1, shuffle=False)

from train_classifier import Classifier, train_model, validate, viewer, visualize, save_distances

model = Classifier(num_classes=len(le.classes_))

#model_name = "noaug-lr-1e-4_epochs-25_batch-32_multiclass"
#model_name = "aug-none_lr-1e-5_epochs-25_batch-32_multiclass"
model_name = "aug-ImgNet_lr-5e-4_epochs-25_batch-64_multiclass"
config = {"name":model_name,
    "max_epochs":25,
    "multi-class":True,
    "Augmentation":"None"}
#model.load_state_dict(torch.load(f"./saves/{model_name}_e-15"))
train_model(model, train_loader, val_loader, name=model_name, num_epochs=config["max_epochs"], lr=5e-4, focal_loss=False)
validate(model, test_loader)
#validate(model, train_loader)

#save_distances(model, all_loader, name2id, id2type, le)
#visualize(model, train_loader, name2id)
exit()
correct = 0
for name, logits, label in viewer(model, test_loader):
    probs = F.softmax(logits, dim=1)
    top_probs = np.argsort(probs).flatten()
    top_names = [(le.classes_[i], round(probs[0,i].item(), 3)) for i in top_probs][::-1]
    true_label = [x for i, x in enumerate(le.classes_) if label[0, i].item() == 1]
    if(label[0,top_probs[-1]] == 1):
        correct += 1
    else:
        print(name, top_names[:3], "/".join(true_label))

print(correct, "/", len(test_loader))

#torch.save(model.state_dict(), "./saves/no_augmentation_e-15")