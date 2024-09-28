import torchvision.transforms as tr
import torch as t
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

def cycle(loader):
    while True:
        for data in loader:
            yield data

def get_data(imagenet_path, fake_images_path):
    # imagenet
    transform_train = tr.Compose([
        tr.Resize(256),
        tr.RandomResizedCrop(224),
        tr.RandomHorizontalFlip(),
        tr.RandomRotation(180),
        tr.RandomVerticalFlip(),
        tr.ToTensor()
        #lambda x: x + 0.01 * t.randn_like(x)
    ])

    # Validation data transformations
    transform_test = tr.Compose([
        tr.Resize(256),
        tr.CenterCrop(224),
        tr.ToTensor()
        #lambda x: x + 0.01 * t.randn_like(x)
    ])


    # Load the training and validation datasets
    dset_train_labeled = ImageFolder(root=imagenet_path, transform=transform_train)
    # make validation
    dset_train, dset_val = t.utils.data.random_split(dset_train_labeled, [len(dset_train_labeled) - 100, 100])
    dset_train_labeled = t.utils.data.Subset(dset_train_labeled, indices=[i for i in range(len(dset_train_labeled)) if
                                                                          i not in dset_val.indices])

    #add paths to image folders

    consistency_images = np.load(fake_images_path, mmap_mode='r')
    gen_con = consistency_images["arr_0"]
    gen_con = t.tensor(gen_con / 255).float().permute(0, 3, 2, 1)
    gen_con_lab = consistency_images["arr_1"]
    gen_con_lab = t.tensor(gen_con_lab).float()

    #duplicate for more fake images of other kinds



    # concatenate all the tensors for the fake images dataset
    fake_images = t.cat((gen_con))
    fake_labels = t.cat((gen_con_lab))
    fake_dataset = t.utils.data.TensorDataset(fake_images, fake_labels)
    print("finished creating fakeset")

    print("loading test set")
    # test dataset
    dset_test = ImageFolder(root=imagenet_path, transform=transform_test)
    print("loading train set")
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=128, shuffle=True, num_workers=1,
                                     drop_last=True, pin_memory=True)
    dload_valid = DataLoader(dset_val, batch_size=128, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    dload_test = DataLoader(dset_test, batch_size=128, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    print("loading fakeset")
    dload_fake = t.utils.data.DataLoader(fake_dataset, batch_size=128, shuffle=True, num_workers=1,
                                             drop_last=True, pin_memory=True)
    dload_fake = cycle(dload_fake)
    print("done")
    return dload_train_labeled, dload_valid, dload_test, dload_fake
