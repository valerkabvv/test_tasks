import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import os
import cv2
import sys
import json
import numpy as np
import albumentations as A
import albumentations.pytorch as AP
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F

PATH = 'model.ckpt'
class classifier(pl.LightningModule):

  def __init__(self, lr = 1e-4):
    super().__init__()
    self.model = EfficientNet.from_pretrained('efficientnet-b4')
    self.model._fc = nn.Linear(1792, 1)
    self.lr = lr

  def forward(self, x):
    return self.model(x)

class test_dataset(Dataset):

  def __init__(self, path_to_folder, transforms):
    self.root = path_to_folder
    self.files = os.listdir(path_to_folder)
    self.transforms = transforms

  def __getitem__(self, ind):
    im = cv2.imread(os.path.join(self.root, self.files[ind]), cv2.COLOR_BGR2RGB)
    return self.transforms(image = im)['image'], self.files[ind]

  def __len__(self):
    return len(self.files)

class max_dim_resize(DualTransform):
    def __init__(self, max_size , interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(max_dim_resize, self).__init__(always_apply, p)
        self.max_size = max_size
        self.interpolation = interpolation
 
    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        a, b, *_ = img.shape
        alpha = float(self.max_size)/max(a,b)
        n_height, n_width = int(a*alpha), int(b*alpha)
        return F.resize(img, height=n_height, width=n_width, interpolation=interpolation)
 
    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox
 
    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)
 
    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")

    
norm_dict = {
    'mean' : np.array([0.485, 0.456, 0.406]),
    'std' : np.array([0.229, 0.224, 0.225])
}

val_transforms = A.Compose([
    max_dim_resize(224),
    A.PadIfNeeded(224,224, border_mode=0),
    AP.transforms.ToTensor(normalize=norm_dict)
])


model = classifier().load_from_checkpoint(PATH)
model = model.cuda()
model = model.eval()

dataset = test_dataset(sys.argv[1], val_transforms)

classes = []
files = []

for x, f in tqdm(DataLoader(dataset, batch_size=256)):
  with torch.no_grad():
    yh = model1(x.cuda())
    classes.append((torch.sigmoid(yh)>0.5).int().cpu().numpy()[:,0])
    files.append(np.array(f))

classes = list(map(lambda x: 'male' if x==1 else 'female', np.hstack(classes)))
files = np.hstack(files)

answer = dict(zip(files, classes))

with open('process_results.json', 'w') as fp:
    json.dump(answer, fp)