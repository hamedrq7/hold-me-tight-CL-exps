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

from utils import get_dataset_loaders
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
PRETRAINED = False
PRETRAINED_PATH = '/home/hamed/EBV/Margins/hold-me-tight-CL-exps/Models/Generated/CL/MNIST/LeNet-center_lr-0.5 alpha-0.1/model.t7'
BATCH_SIZE = 128

# Load a model
model = ResNet18(zero_bias=(METHOD=='CL'))  # check inside the model_class.cifar10 package for other network options

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


# If pretrained
if PRETRAINED:
    print('---> Working on a pretrained network')
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location='cpu'))
    model = model.to(DEVICE)
    model.eval()

# If not pretrained, then train it
if not PRETRAINED:

    EPOCHS = 30
    MAX_LR = 0.21
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    opt = torch.optim.SGD(model.parameters(), lr=MAX_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_fun = nn.CrossEntropyLoss()
    lr_schedule = lambda t: np.interp([t], [0, EPOCHS * 2 // 5, EPOCHS], [0, MAX_LR, 0])[0]  # Triangular (cyclic) learning rate schedule

    if METHOD == 'CE': 
        SAVE_TRAIN_DIR = TREE_ROOT + f'Models/Generated/{DATASET}/{model.__class__.__name__}/{METHOD}/'
    elif METHOD == 'CL': 
        SAVE_TRAIN_DIR = TREE_ROOT + f'Models/Generated/{DATASET}/{model.__class__.__name__}/{METHOD}/center_lr-{center_lr} alpha-{alpha} epochs-{EPOCHS}/'
    os.makedirs(SAVE_TRAIN_DIR, exist_ok=True)
    
    t0 = time.time()
    model = model.to(DEVICE)
    from torchinfo import summary
    summary(model, (1, 3, 32, 32))

    if METHOD == 'CE':
        model = train(model, trans, trainloader, testloader, EPOCHS, opt, loss_fun, lr_schedule, SAVE_TRAIN_DIR)
    elif METHOD == 'CL': 
        model = train_cl(model, DEVICE, trans, trainloader, testloader, EPOCHS, opt, loss_fun, lr_schedule, SAVE_TRAIN_DIR, 
                         center_lr=center_lr, alpha=alpha)
        
    print('---> Training is done! Elapsed time: %.5f minutes\n' % ((time.time() - t0) / 60.))

RESULTS_DIR = os.path.dirname(PRETRAINED_PATH) if PRETRAINED else SAVE_TRAIN_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

#####################################
# Compute Robustness using DeepFool #
#####################################
from utils import deepfool, get_eval

from utils import get_eval
NUM_SAMPLES_EVAL = 100
eval_dataset, eval_loader, NUM_SAMPLES_EVAL = get_eval(testset=testset, num_samples=NUM_SAMPLES_EVAL, batch_size=128)

l2_norms = []
linf_norms = []

for img, lbl in eval_loader:
    img = trans(img.to(DEVICE))[0, :, :, :] # 
    lbl = lbl.to(DEVICE)
    minimal_perturbation, number_iterations, true_label, new_label, perturbed_image = deepfool(
        image=img, 
        net=model,
        num_classes=10,
        overshoot=0.02,
        max_iter=50
    )
    minimal_perturbation = minimal_perturbation.squeeze()
    
    # print('max', img.max())
    # print('min', img.min())
    # # print(minimal_perturbation)
    # print('l2 minimal_perturbation', np.linalg.norm(minimal_perturbation))
    # print('linf minimal_perturbation', np.max(np.abs(minimal_perturbation)))
    # print('number_iterations', number_iterations)
    # print()

    l2_norm = np.linalg.norm(minimal_perturbation)
    linf_norm = np.max(np.abs(minimal_perturbation))

    l2_norms.append(l2_norm)
    linf_norms.append(linf_norm)

from utils import plot_norms
plot_norms(l2_norms, 'l-2', 
           title=f'{METHOD} Histogram of $L_2$ Norms of Minimal Perturbations\nMean: {np.mean(l2_norms): .4f}', 
           path_to_save=RESULTS_DIR)

plot_norms(linf_norms, 'l-inf', 
           title=f'{METHOD} Histogram of $L_\infty$ Norms of Minimal Perturbations\nMean: {np.mean(linf_norms): .4f}', 
           path_to_save=RESULTS_DIR)

##################################
# Compute margin along subspaces #
##################################
# Select the data samples for evaluation
from utils import get_eval
NUM_SAMPLES_EVAL = 100
eval_dataset, eval_loader, NUM_SAMPLES_EVAL = get_eval(testset=testset, num_samples=NUM_SAMPLES_EVAL, batch_size=128)

# Create a list of subspaces to evaluate the margin on
SUBSPACE_DIM = 8
DIM = 32
SUBSPACE_STEP = 2

subspace_list = generate_subspace_list(SUBSPACE_DIM, DIM, SUBSPACE_STEP, channels=3)

# Compute the margin using subspace DeepFool and save the results
margins = compute_margin_distribution(model, trans, eval_loader, subspace_list, RESULTS_DIR + 'margins.npy')

from graphics import swarmplot
swarmplot(margins, name = f'{RESULTS_DIR}/{METHOD}-{model.__class__.__name__}',color='tab:blue')
