import torchvision.transforms as transforms
from autoaugment import ImageNetPolicy
import config as cf

def transform_training():

    transform_train = transforms.Compose([
        transforms.Resize(cf.resize),
        transforms.CenterCrop(cf.crop_size),
        transforms.RandomResizedCrop(cf.resize),
        transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])  # meanstd transformation

    return transform_train

def transform_testing():

    transform_test = transforms.Compose([
        transforms.Resize(cf.resize),
        transforms.CenterCrop(cf.crop_size),
        transforms.RandomResizedCrop(cf.resize),
        transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform_test


def transform_testing_vgg():

    transform_test = transforms.Compose([
        #transforms.Resize((cf.resize, cf.resize)),
        #transforms.RandomCrop(224, padding=4),
        # transforms.RandomHorizontalFlip(),
        #CIFAR10Policy(),
       #Dog_BreedPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform_test