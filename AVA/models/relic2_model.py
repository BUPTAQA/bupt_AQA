import sys
sys.path.append('..')
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from models.mv2 import cat_net
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import option
opt = option.init()
opt.device = torch.device("cuda:{}".format(opt.gpu_id))
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

### ???
def SelfAttentionMap(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        base_model = cat_net()

        self.base_model = base_model
        for p in self.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(8,64)
        self.relu = nn.Tanh()
        # 𝛼 and 𝛽 denote the weighting factors of composition features and miscellaneous aesthetic features
        self.fc1 = nn.Linear(64,2)
        # self.sm = nn.Softmax(dim=1)
        self.sm = nn.Sigmoid()


        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(3681, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1, x2 = self.base_model(x)
        x1_max = torch.max(x1,dim=1)[0].unsqueeze(1)
        x1_min = torch.min(x1,dim=1)[0].unsqueeze(1)
        x1_mean = torch.mean(x1,dim=1).unsqueeze(1)
        x1_std = torch.std(x1,dim=1).unsqueeze(1)
        x1_in = torch.cat([x1_max,x1_min,x1_mean,x1_std],1)

        x2_max = torch.max(x2,dim=1)[0].unsqueeze(1)
        x2_min = torch.min(x2,dim=1)[0].unsqueeze(1)
        x2_mean = torch.mean(x2,dim=1).unsqueeze(1)
        x2_std = torch.std(x2,dim=1).unsqueeze(1)
        x2_in = torch.cat([x2_max,x2_min,x2_mean,x2_std],1)

        x_in = torch.cat([x1_in,x2_in],1)
        # two weighting factors.
        x_out =self.sm(self.fc1(self.relu(self.fc(x_in))))

        x1 = x1 * torch.unsqueeze(x_out[:, 0], 1)
        x2 = x2 * torch.unsqueeze(x_out[:, 1], 1)
        x = torch.cat([x1,x2],1)
        x = self.head(x)

        return x

# transform picture
def TransformPicture(x):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
    return transform(x)

def get_score(opt,y_pred):
    w = torch.from_numpy(np.linspace(1,10, 10))
    w = w.type(torch.FloatTensor).cpu()
    # w = w.to(opt.device)
    y_pred.cpu()

    w_batch = w.repeat(y_pred.size(0), 1)
    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

if __name__=='__main__':
    model = NIMA()
    model.eval()
    model.load_state_dict(torch.load('./pretrain_model/relic2_model.pth', map_location='cuda:0'))
    # print(model)

    # give a random picture
    # image = torch.rand((1,3,224,224))

    # get a picture from path
    image_path = './'
    # image_path = os.path.join(image_path, f'150.jpg')
    image_path = os.path.join(image_path, opt.path_to_images)
    image = default_loader(image_path)
    plt.imshow(image)
    plt.show()
    image = TransformPicture(image)
    image = torch.unsqueeze(image, 0)

    out = model(image)
    score,_ = get_score(opt, out)
    num_score = round(score.item(), 2)
    print('The aesthetic score is:'+ str(num_score) + '.\n')
    print('It {judge} a good photo.'.format(judge='is' if score > 5 else 'is not'))

    # write to file
    filename = 'score.txt'
    with open(filename, 'w') as f:
        f.write('The aesthetic score is:'+ str(num_score) + '.\n')
        f.write('It {judge} a good photo.'.format(judge='is' if score > 5 else 'is not'))
