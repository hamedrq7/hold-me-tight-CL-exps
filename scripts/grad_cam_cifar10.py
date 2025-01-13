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
from model_classes.cifar10 import ResNet18  # check inside the model_class.cifar10 package for other network options
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
center_lr = 0.5
alpha = 0.01

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET = 'CIFAR10'
PRETRAINED_PATH = '/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/CE/model.t7'
# PRETRAINED_PATH = '/home/ramin/Margin_analysis/hold-me-tight-CL-exps/Models/Generated/CIFAR10/ResNet18/CL/center_lr-0.5 alpha-0.01 epochs-30/model.t7'

BATCH_SIZE = 128
RESULTS_DIR = os.path.dirname(PRETRAINED_PATH)

# Load a model
model = ResNet18(zero_bias=(METHOD=='CL'))  # check inside the model_class.cifar10 package for other network options
print('---> Working on a pretrained network')
model.load_state_dict(torch.load(PRETRAINED_PATH, map_location='cpu'))
model = model.to(DEVICE)
model.eval()

#############################
# Dataset paths and loaders #
#############################

# Specify the path of the dataset. For MNIST and CIFAR-10 the train and validation paths can be the same.
# For ImageNet, please specify to proper train and validation paths.
# DATASET_DIR = {'train': os.path.join(TREE_ROOT, '/home/hamed/Storage/LDA-FUM HDD/data/CIFAR10'),
#                'val': os.path.join(TREE_ROOT, '/home/hamed/Storage/LDA-FUM HDD/data/CIFAR10')
#                }
DATASET_DIR = {'train': os.path.join(TREE_ROOT, '/home/ramin/Robustness/LDA-FUM-TEMP/data/CIFAR10'),
               'val': os.path.join(TREE_ROOT, '/home/ramin/Robustness/LDA-FUM-TEMP/data/CIFAR10')
               }

os.makedirs(DATASET_DIR['train'], exist_ok=True)
os.makedirs(DATASET_DIR['val'], exist_ok=True)

# Load the data
trainloader, testloader, trainset, testset, mean, std = get_dataset_loaders(DATASET, DATASET_DIR, BATCH_SIZE)

# Normalization layer
trans = TransformLayer(mean=mean, std=std)

def undo_transform(transformed_images, mean, std):
    mean = mean.squeeze().cpu()  # Remove batch and device dimensions
    std = std.squeeze().cpu()
    # Reverse the transformation
    if isinstance(transformed_images, np.ndarray):
        transformed_images = torch.tensor(transformed_images)

    # Reverse the transformation
    images = transformed_images * std + mean
    return images.numpy()

def grad_cam(
    model: nn.Module,
    loader,
    device,
    name_to_save: str,
    save_path: str,
):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import cv2

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
    
    input, labels = get_samples(loader, 10, )
    input = torch.from_numpy(input).to(device)
    labels = torch.from_numpy(labels).to(device)


    def save_cam_results_cifar(cam_image, heatmap, save_path):
        # cam_image.shape is [32, 32, 3]
        # heatmap.shape is [1, 32, 32]
        cam_image = undo_transform(cam_image, mean=mean, std=std)
        cam_image = np.uint8(cam_image * 255)
        heatmap = (heatmap - np.min(heatmap)) / (
            np.max(heatmap) - np.min(heatmap)
        )
        heatmap = np.uint8(heatmap.squeeze() * 255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        concatenated_image = np.concatenate((cam_image, heatmap), axis=1)
        cv2.imwrite(save_path, concatenated_image)
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
        
    model = single_output_wrapper(model)
    model.eval()

    target_layers = model.grad_layer

    make_dir(f'{save_path}/grad_cam')
    i = 0
    for image in input:
        image = image.unsqueeze(0)
        cam_image = image.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
        image = image.to(device)

        cam = GradCAM(model=model, target_layers=target_layers)
        heatmap = cam(input_tensor=image, targets=None)

        path = f"{save_path}/grad_cam/gradcam_img_{i}.jpg"
        save_cam_results_cifar(cam_image, heatmap, path)
        i += 1

    concat_cam_results(f"{save_path}/grad_cam", f"{name_to_save}")

grad_cam(model=model, device=DEVICE, loader=trainloader, name_to_save='grad_cam', save_path=RESULTS_DIR)
