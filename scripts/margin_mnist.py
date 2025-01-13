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
center_lr = 0.5
alpha = 0.01

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET = 'MNIST'
PRETRAINED = False
PRETRAINED_PATH = '/home/hamed/EBV/Margins/hold-me-tight-CL-exps/Models/Generated/CL/MNIST/LeNet-center_lr-0.5 alpha-0.1/model.t7'
BATCH_SIZE = 128

# Load a model
# model = LeNet(zero_bias=(METHOD=='CL')) # check inside the model_class.mnist package for other network options
model = MNIST_TRADES(zero_bias=(METHOD=='CL')) # ResNet18() #   # check inside the model_class.mnist package for other network options
# model = ResNet18()

#############################
# Dataset paths and loaders #
#############################

# Specify the path of the dataset. For MNIST and CIFAR-10 the train and validation paths can be the same.
# For ImageNet, please specify to proper train and validation paths.
# DATASET_DIR = {'train': os.path.join(TREE_ROOT, '/home/hamed/Storage/LDA-FUM HDD/data/MNIST'),
#                'val': os.path.join(TREE_ROOT, '/home/hamed/Storage/LDA-FUM HDD/data/MNIST')
#                }
DATASET_DIR = {'train': os.path.join(TREE_ROOT, '/home/ramin/Robustness/LDA-FUM-TEMP/data/MNIST'),
               'val': os.path.join(TREE_ROOT, '/home/ramin/Robustness/LDA-FUM-TEMP/data/MNIST')
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

    ### Original paper HyperParams
    # TRADES_SETTING=False
    # EPOCHS = 30
    # MAX_LR = 0.21
    # MOMENTUM = 0.9
    # WEIGHT_DECAY = 5e-4
    # opt = torch.optim.SGD(model.parameters(), lr=MAX_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # lr_schedule = lambda t: np.interp([t], [0, EPOCHS * 2 // 5, EPOCHS], [0, MAX_LR, 0])[0]  # Triangular (cyclic) learning rate schedule

    ### TRADES setting
    TRADES_SETTING=True
    EPOCHS = 100
    MAX_LR = 0.01
    WEIGHT_DECAY = 0.0
    MOMENTUM = 0.9
    gamma = 0.1
    milestones = [55, 75, 90]
    lr_schedule = lambda t: MAX_LR * (gamma ** sum([int(t) >= milestone for milestone in milestones]))
    opt = torch.optim.SGD(model.parameters(), lr=MAX_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    loss_fun = nn.CrossEntropyLoss()

    if METHOD == 'CE': 
        SAVE_TRAIN_DIR = TREE_ROOT + f'Models/Generated/{DATASET}/{model.__class__.__name__}-tradesSetting_{TRADES_SETTING}/{METHOD}/'
    elif METHOD == 'CL': 
        SAVE_TRAIN_DIR = TREE_ROOT + f'Models/Generated/{DATASET}/{model.__class__.__name__}-tradesSetting_{TRADES_SETTING}/{METHOD}/center_lr-{center_lr} alpha-{alpha} epochs-{EPOCHS}/'
    os.makedirs(SAVE_TRAIN_DIR, exist_ok=True)
    
    t0 = time.time()
    model = model.to(DEVICE)
    from torchinfo import summary
    summary(model, (1, 1, 28, 28))

    if METHOD == 'CE':
        model = train(model, trans, trainloader, testloader, EPOCHS, opt, loss_fun, lr_schedule, SAVE_TRAIN_DIR)
    elif METHOD == 'CL': 
        model = train_cl(model, DEVICE, trans, trainloader, testloader, EPOCHS, opt, loss_fun, lr_schedule, SAVE_TRAIN_DIR, 
                         center_lr=center_lr, alpha=alpha)

    print('---> Training is done! Elapsed time: %.5f minutes\n' % ((time.time() - t0) / 60.))


##################################
# Compute margin along subspaces #
##################################

# Create a list of subspaces to evaluate the margin on
SUBSPACE_DIM = 8
DIM = 28
SUBSPACE_STEP = 1

subspace_list = generate_subspace_list(SUBSPACE_DIM, DIM, SUBSPACE_STEP, channels=1)

# Select the data samples for evaluation
NUM_SAMPLES_EVAL = 100
indices = np.random.choice(len(testset), NUM_SAMPLES_EVAL, replace=False)

eval_dataset = torch.utils.data.Subset(testset, indices[:NUM_SAMPLES_EVAL])
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2, pin_memory=True if DEVICE == 'cuda' else False)

RESULTS_DIR = os.path.dirname(PRETRAINED_PATH) if PRETRAINED else SAVE_TRAIN_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

# Compute the margin using subspace DeepFool and save the results
margins = compute_margin_distribution(model, trans, eval_loader, subspace_list, RESULTS_DIR + 'margins.npy')

from graphics import swarmplot
swarmplot(margins, name = f'{RESULTS_DIR}/{METHOD}-{model.__class__.__name__}',color='tab:blue')