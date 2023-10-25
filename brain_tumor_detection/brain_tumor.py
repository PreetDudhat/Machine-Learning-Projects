import cv2
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch._C import Node
from torch.nn.modules import module
from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image


def save_image():
    for i in os.listdir('data'):
        for j, image in enumerate(os.listdir('data/{}'.format(i))):
            image_path = 'data/{}/'.format(i) + image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            resize_data(image, j, i)


def resize_data(image, index, folder_name):
    sample_dir = 'resized_images'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    path = os.path.join(sample_dir, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)

    resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path + '/{}'.format(index) + '.jpg', resized_image)


save_image()

dataset = ImageFolder('./resized_images', transform=tt.ToTensor())
train_indices = np.random.permutation(len(dataset))
train_ds = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset=dataset, batch_size=10, sampler=train_ds)


def mean(loader):
    nimages = 0
    mean = 0.
    std = 0.
    for batch, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    # Final step
    mean /= nimages
    std /= nimages

    return [tuple(mean.tolist()), tuple(std.tolist())]


stats = mean(train_dl)
print(stats)
dataset = ImageFolder('./resized_images', transform=tt.Compose([tt.RandomCrop(
    512, padding=10, padding_mode='reflect'), tt.RandomHorizontalFlip(), tt.ToTensor(), tt.Normalize(*stats)]))

def split_indices(n, val_pct=0.1, seed=99):
    n_val = int(n * val_pct)
    np.random.seed(seed)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


train_indices, val_indices = split_indices(len(dataset), 0.4, 42)

train_ds = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset=dataset, batch_size=10, sampler=train_ds)

val_ds = SubsetRandomSampler(val_indices)
val_dl = DataLoader(dataset=dataset, batch_size=10, sampler=val_ds)

model = nn.Sequential(                                  # bs X 3 X 512 X 512
    nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),   # bs X 16 X 512 X 512
    nn.ReLU(),
    nn.BatchNorm2d(16),                                          # bs X 16 X 512 X 512
    nn.MaxPool2d(2,2),                                  # bs X 16 X 256 X 256

    nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),  # bs X 32 X 256 X 256
    nn.ReLU(),     
    nn.BatchNorm2d(32),                                     # bs X 32 X 256 X 256
    nn.MaxPool2d(2,2),                                  # bs X 32 X 128 X 128

    nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),  # bs X 32 X 128 X 128
    nn.ReLU(),  
    nn.BatchNorm2d(64),                                        # bs X 64 X 128 X 128
    nn.MaxPool2d(2,2),                                  # bs X 64 X 64 X 64

    nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),  # bs X 64 X 64 X 64
    nn.ReLU(),        
    nn.BatchNorm2d(128),                                  # bs X 128 X 64 X 64
    nn.MaxPool2d(2,2),                                  # bs X 128 X 32 X 32

    nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),  # bs X 128 X 32 X 32
    nn.ReLU(),    
    nn.BatchNorm2d(256),                                      # bs X 256 X 32 X 32
    nn.MaxPool2d(2,2),                                  # bs X 256 X 16 X 16

    nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),  # bs X 256 X 16 X 16
    nn.ReLU(),         
    nn.BatchNorm2d(512),                                 # bs X 512 X 16 X 16
    nn.MaxPool2d(2,2),    

    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),  # bs X 512 X 8 X 8
    nn.ReLU(),      
    nn.BatchNorm2d(512),                                    # bs X 16 X 8 X 8
    nn.MaxPool2d(2,2),    

    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),  # bs X 512 X 4 X 4
    nn.ReLU(),     
    nn.BatchNorm2d(512),                                     # bs X 512 X 4 X 4
    nn.MaxPool2d(2,2),    

    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),  # bs X 512 X 2 X 2
    nn.ReLU(),         
    nn.BatchNorm2d(512),                                 # bs X 512 X 2 X 2
    nn.MaxPool2d(2,2),    

    nn.Flatten(),                                       # bs X 512
    nn.Linear(512,256), 
    nn.Linear(256,128),   
    nn.Linear(128,64),   
    nn.Linear(64,2)            
)


# loss function is cross_entropy in which softmax calculations are included 
loss_func = F.cross_entropy

# this below 2 functions are only for calculating loss
# calculation loss across batch
def loss_batch(model,loss_func,xb,yb,opt=None,metric=None):
    preds = model(xb)
    loss = loss_func(preds,yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds,yb)

    return loss.item(),len(xb),metric_result

# calculate avg loss of every loss across batchs
def evaluate(model,loss_func,valid_dl,metric=None):
    with torch.no_grad():
        # pass each batch through the model
        results = [loss_batch(model,loss_func,xb,yb,metric=metric) for xb,yb in valid_dl]
        # separate losses,batch size,metric result
        losses,nums,metrics = zip(*results)
        m = []
        for i,j in zip(list(losses),list(nums)):
            m.append(i*j)
        m = np.array(m)
        total = np.sum(nums)
        avg_loss = np.sum(m)/total
        avg_metric = None
        if metric is not None:
            m1 = []
            for i,j in zip(list(metrics),list(nums)):
                m1.append(i*j)
            m1 = np.array(m1)
            avg_metric = np.sum(m1)/total
    return avg_loss,total,avg_metric


# accuracy
def accuracy(outputs,labels):
    _, preds = torch.max(outputs,dim=1)
    return torch.sum(preds == labels).item()/len(preds)

# val_loss,total,val_acc = evaluate(model,loss_func,val_loader,metric=accuracy)
# print(val_loss,total,val_acc)

# optimizer
# opt = torch.optim.SGD(model.parameters(),lr=0.001)
opt = torch.optim.Adam(model.parameters(),lr=0.005)

# training model
def fit(model,train_loader,val_loader,opt,loss_func,epochs,metrics=None):
    for epoch in range(epochs):
        # batch wise training
        model.train()
        for xb,yb in train_loader:
            loss,_,_ = loss_batch(model,loss_func,xb,yb,opt)
        
        # evaluation
        model.eval()
        val_loss,total,val_acc = evaluate(model,loss_func,val_loader,metrics)

        if metrics is None:
            print('Epoch[{}/{}],Loss:{:.4f}'.format(epoch+1,epochs,val_loss))
        else:
            print('Epoch[{}/{}],Loss:{:.4f},Accuracy:{:.4f}'.format(epoch+1,epochs,val_loss,val_acc))



fit(model,train_dl,val_dl,opt,loss_func,0,metrics=accuracy)

# #save the model
# torch.save(model.state_dict(),'cifar10_classification.pth')

# # prediction
def predict(img,model):
    yb = img.unsqueeze(0)
    yb = model(yb)
    _,preds = torch.max(yb,dim=1)
    return dataset.classes[preds[0].item()]

convert_tensor = transforms.ToTensor()
for i in os.listdir('test'):
    img = Image.open("test/{}".format(i))
    image = convert_tensor(img)
    print(image.shape,i)
    p = predict(image,model)
print('label : {},Predicted : {}'.format(dataset.classes[1],p))