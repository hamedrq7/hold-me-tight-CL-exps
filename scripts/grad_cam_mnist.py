import numpy as np
import torch
import torch.nn as nn
import os
import time

import sys
# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user

from utils import get_dataset_loaders, make_dir
from utils import train, train_cl
from utils import generate_subspace_list
from utils import compute_margin_distribution
from model_classes import TransformLayer
from model_classes.mnist import LeNet, ResNet18, MNIST_TRADES  # check inside the model_class.mnist package for other network options
import random 

seed = 111
# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



TREE_ROOT = ''
METHOD = 'CE' # 'CL'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET = 'MNIST'
PRETRAINED_PATH = '/home/hamed/EBV/Margins/hold-me-tight-CL-exps/Models/Generated/MNIST/MNIST_TRADES-tradesSetting_True/CL/center_lr-0.5 alpha-0.1 epochs-25/model.t7'
PRETRAINED_PATH = '/home/hamed/EBV/Margins/hold-me-tight-CL-exps/Models/Generated/MNIST/MNIST_TRADES-tradesSetting_True/CE/model.t7'
BATCH_SIZE = 128
RESULTS_DIR = os.path.dirname(PRETRAINED_PATH)

# Load a model
# model = LeNet(zero_bias=(METHOD=='CL')) # check inside the model_class.mnist package for other network options
model = MNIST_TRADES(zero_bias=(METHOD=='CL')) # ResNet18() #   # check inside the model_class.mnist package for other network options
# model = ResNet18()

print('---> Working on a pretrained network')
model.load_state_dict(torch.load(PRETRAINED_PATH, map_location='cpu'))
model = model.to(DEVICE)
model.eval()

#############################
# Dataset paths and loaders #
#############################

# Specify the path of the dataset. For MNIST and CIFAR-10 the train and validation paths can be the same.
# For ImageNet, please specify to proper train and validation paths.
DATASET_DIR = {'train': os.path.join(TREE_ROOT, '/home/hamed/Storage/LDA-FUM HDD/data/MNIST'),
               'val': os.path.join(TREE_ROOT, '/home/hamed/Storage/LDA-FUM HDD/data/MNIST')
               }
# DATASET_DIR = {'train': os.path.join(TREE_ROOT, '/home/ramin/Robustness/LDA-FUM-TEMP/data/MNIST'),
#                'val': os.path.join(TREE_ROOT, '/home/ramin/Robustness/LDA-FUM-TEMP/data/MNIST')
#                }
os.makedirs(DATASET_DIR['train'], exist_ok=True)
os.makedirs(DATASET_DIR['val'], exist_ok=True)

# Load the data
trainloader, testloader, trainset, testset, mean, std = get_dataset_loaders(DATASET, DATASET_DIR, BATCH_SIZE)

# Normalization layer
trans = TransformLayer(mean=mean, std=std)

def grad_cam(
    model: nn.Module,
    loader,
    device,
    name_to_save: str,
    save_path: str,
):
    
    def get_samples(dataloader, num_samples: int, ) -> np.ndarray:
        samples = [[] for _ in range(10)]
        samples_lbl = [[] for _ in range(10)]

        for batch_indx, (images, labels) in enumerate(dataloader):
            # images are normalized using mean = (0.0, ) and std = (1.0, ),
            # so images have been normalized using: image = image - mean / std
            # to plot images we have to undo the normalization
            # images = (images - 0.1307) / 0.3081
            images = trans(images.to(device)).cpu()

            for img_indx, curr_image in enumerate(images):

                if len(samples[labels[img_indx]]) < num_samples:
                    samples[labels[img_indx]].append(curr_image.numpy())
                    samples_lbl[labels[img_indx]].append(labels[img_indx])

        # convert samples to numpy array
        samples = np.array(samples)
        samples_lbl = np.array(samples_lbl)
        samples = samples.reshape(samples.shape[0] * samples.shape[1], *samples.shape[2:])
        samples_lbl = samples_lbl.reshape(
            samples_lbl.shape[0] * samples_lbl.shape[1], *samples_lbl.shape[2:]
        )
        return samples, samples_lbl

    def save_cam_results(cam_image, heatmap, save_path):
        heatmap = heatmap.squeeze()

        img1 = cv2.normalize(
            cam_image,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        img2 = cv2.normalize(
            heatmap,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        img2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)
        img1 = cv2.merge([img1] * 3)

        vis = cv2.hconcat([img1, img2])
        cv2.imwrite(save_path, vis)
        return

    def concat_cam_results(save_path, name_to_save):
        image_files = sorted(
            [f for f in os.listdir(save_path) if f.startswith("gradcam")],
            key=lambda x: int("".join(filter(str.isdigit, x))),
        )

        canvas_height = 1300
        canvas_width = 2300
        cell_height = 100
        cell_width = 200
        spacing = 20

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for i, image_file in enumerate(image_files):
            row = i // 10
            col = i % 10
            x_offset = col * (cell_width + spacing) + 100
            y_offset = row * (cell_height + spacing) + 100
            image = cv2.imread(os.path.join(save_path, image_file))
            image = cv2.resize(image, (cell_width, cell_height))
            canvas[
                y_offset : y_offset + cell_height, x_offset : x_offset + cell_width
            ] = image

        cv2.imwrite(f"{save_path}/final_image_{name_to_save}.jpg", canvas)
        return

    class single_output_wrapper(nn.Module):
        def __init__(self, model, ):
            super(single_output_wrapper, self).__init__()
            self.model = model
            self.grad_layer = model.grad_layer
            
        def forward(self, x):
            _, output2 = self.model(x)
            return output2
    
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import cv2

    input, labels = get_samples(loader, 10, )
    input = torch.from_numpy(input).to(device)
    labels = torch.from_numpy(labels).to(device)

    model = single_output_wrapper(model)
    model.eval()

    target_layers = model.grad_layer

    make_dir(f'{save_path}/grad_cam')
    i = 0
    for image in input:
        image = image[None, :]
        cam_image = image.detach().cpu().numpy().squeeze()
        image = image.to(device)

        cam = GradCAM(model=model, target_layers=target_layers)
        heatmap = cam(input_tensor=image, targets=None)

        path = f"{save_path}/grad_cam/gradcam_img_{i}.jpg"
        save_cam_results(cam_image, heatmap, path)
        i += 1

    concat_cam_results(f"{save_path}/grad_cam", f"{name_to_save}")

grad_cam(model=model, device=DEVICE, loader=trainloader, name_to_save='grad_cam', save_path=RESULTS_DIR)
