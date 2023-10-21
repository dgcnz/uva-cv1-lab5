from torchvision import transforms

default_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

def sized_transform(size: int = 224):
    return {
        "train": transforms.Compose(
        [
            transforms.Resize(size + 2),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomPerspective(),
            transforms.RandomAffine(15),
            transforms.RandomErasing(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
        
        "val": transforms.Compose(
        [
            transforms.Resize(size + 2),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    }