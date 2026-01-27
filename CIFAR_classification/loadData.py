"""
===================================================================================
    Source Name   : DataLoaders.py
    Description   : Function that creates the train, validation, and test data loaders for CIFAR-10
===================================================================================
"""
# %% import dependencies
import torch
import torchvision
import torchvision.transforms as transforms

# %% create dataloader
def load(args):
    batch_size = args.batch_size
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    train_size = int(0.9 * len(trainset))   # 90% train, 10% validation
    val_size = len(trainset) - train_size
    train_data, val_data = torch.utils.data.random_split(trainset, [train_size, val_size])
        

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, drop_last=True)

    
    return train_loader, val_loader, test_loader
