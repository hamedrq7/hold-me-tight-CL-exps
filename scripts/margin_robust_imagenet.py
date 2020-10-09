import numpy as np
import torch
import os

from utils import get_dataset_loaders
from utils import generate_subspace_list
from utils import compute_margin_distribution
from model_classes import TransformLayer
from torchvision.models import resnet50


TREE_ROOT = './'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET = 'ImageNet'
BATCH_SIZE = 128
PRETRAINED_PATH = '../Models/Pretrained/ImageNet_robust/ResNet50/model.t7'

#############################
# Dataset paths and loaders #
#############################
# Specify the path of the dataset. For MNIST and CIFAR-10 the train and validation paths can be the same.
# For ImageNet, please specify to proper train and validation paths.
DATASET_DIR = {'train': 'path-to-ILSVRC2012-train-data',
               'val': 'path-to-ILSVRC2012-val-data'
               }

# Load the data
trainloader, testloader, trainset, testset, mean, std = get_dataset_loaders(DATASET, DATASET_DIR, BATCH_SIZE)


####################
# Select a Network #
####################

# Normalization layer
trans = TransformLayer(mean=mean, std=std)

# Load a pretrained model
model = resnet50()
model.load_state_dict(torch.load(PRETRAINED_PATH, map_location='cpu'))
model = model.to(DEVICE)
model.eval()


##################################
# Compute margin along subspaces #
##################################

# Create a list of subspaces to evaluate the margin on
SUBSPACE_DIM = 16
DIM = 224
SUBSPACE_STEP = 16

subspace_list = generate_subspace_list(SUBSPACE_DIM, DIM, SUBSPACE_STEP, channels=3)

# Select the data samples for evaluation
NUM_SAMPLES_EVAL = 100
indices = np.random.choice(len(testset), NUM_SAMPLES_EVAL, replace=False)

eval_dataset = torch.utils.data.Subset(testset, indices[:NUM_SAMPLES_EVAL])
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2, pin_memory=True if DEVICE == 'cuda' else False)

# Compute the margin using subspace DeepFool and save the results
RESULTS_DIR = os.path.join(TREE_ROOT, '../Results/margin_%s_robust/%s/' % (DATASET, model.__class__.__name__))
os.makedirs(RESULTS_DIR, exist_ok=True)

margins = compute_margin_distribution(model, trans, eval_loader, subspace_list, RESULTS_DIR + 'margins.npy')
