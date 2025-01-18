import torch 
import numpy as np 
import torch.nn as nn 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import scipy 
import os 
import sys
# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user

import torch.nn.functional as F

from utils import get_dataset_loaders

def make_dir(path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)


def get_angle_origin(vec): 
    
    vectors = np.array([vec, [1, 0]])
 
    dist = scipy.spatial.distance.pdist(vectors, 'cosine')
 
    angle = np.rad2deg(np.arccos(1 - dist))
    coss = 1 - dist
    # print(coss)
    res = angle[0] 
    if vec[1] < 0: 
        res = 360 - res

    return res, coss[0]

def get_boundaries(w, line_length):

    angle_arr = []
    cos_arr = []

    # print(w)
    # print(w.shape)

    for i in range(10):
        w_i = w[i, :]
        angle, cos = get_angle_origin(w_i)
        angle_arr.append(angle)
        cos_arr.append(cos)

    angle_arr = np.array(angle_arr)
    cos_arr = np.array(cos_arr)

    angl_perm = np.argsort(angle_arr)
    # print(angl_perm)

    def temp2(w_1_idx, w_2_idx, alpha): 
        # print('--', w_1_idx, w_2_idx)
        angle1 = angle_arr[w_1_idx]
        angle2 = angle_arr[w_2_idx]
        # print('angles...', angle1, angle2)

        # print('alpha', alpha)
        is_class1 = []
        angle_label = []
        for counter in np.linspace(0, 360, 3601):
            t = angle1 + counter % 360
            theta1 = (t - angle1) % 360
            theta2 = (t - angle2) % 360 
            
            score1 = np.linalg.norm(w[w_1_idx, :]) * np.cos(np.deg2rad(theta1))
            score2 = np.linalg.norm(w[w_2_idx, :]) * np.cos(np.deg2rad(theta2))
            # print(f'{theta1: .3f} vs {(theta2): .3f}', f'score1: {score1:.3f}', f'- score2: {score2:.3f}')
            
            is_class1.append([np.cos(np.deg2rad(theta1+angle_arr[w_1_idx])), np.sin(np.deg2rad(theta1+angle_arr[w_1_idx])), float(score1 >= score2)])
            angle_label.append([theta1+angle_arr[w_1_idx], float(score1 >= score2)])

        angle_label = np.array(angle_label)

        angle_label = angle_label % 360
        angle_label = angle_label[np.argsort(angle_label[:, 0], axis=0)]
        
        # print(angle_label)
        # exit()
        return angle_label, is_class1

    def get_st_end(R): 
        mask = R[:, 1] == 1.0
        start_indices = np.where(mask & ~np.roll(mask, 1))[0]
        end_indices = np.where(mask & ~np.roll(mask, -1))[0]
        if mask[0] and mask[-1]:
            end_indices[0] = end_indices[-1]

        # print("Indices where consecutive 1's start:", start_indices)
        # print("Indices where consecutive 1's end:", end_indices)
        return start_indices[0], end_indices[0]

    def temp(w_idx, line_length, ax = None):

        w_angle_idx = np.where(angl_perm==w_idx)[0][0]  
        prev_idx = angl_perm[(w_angle_idx-1)%10]
        next_idx = angl_perm[(w_angle_idx+1)%10]

        # print(f'comparing class {w_idx} with next({next_idx}) and prev({prev_idx})')
        # w_i    = w[w_idx]
        # w_next = w[next_idx]
        # w_last = w[prev_idx]

        angle_from_next = (angle_arr[next_idx]-angle_arr[w_idx]) % 360
        angle_from_prev = (angle_arr[w_idx]   -angle_arr[prev_idx]) % 360

        angle_label_prev, points1 = temp2(w_idx, prev_idx, angle_from_prev)
        angle_label_next, points2 = temp2(w_idx, next_idx, angle_from_next)
        
        # print(angle_label_prev)
        # print(angle_label_next)

        points1 = np.array(points1)
        points2 = np.array(points2)

        intersect_angles = angle_label_prev
        intersect_angles[:, 1] = intersect_angles[:, 1] * angle_label_next[:, 1]
        st, end = get_st_end(intersect_angles)
        # intersect_angles = intersect_angles[intersect_angles[:, 1] == 1.0]
        
        angle1 = intersect_angles[st][0]
        angle2 = intersect_angles[end][0]
        
        # print('ang1, ang2:', angle1, angle2)
        
        x1 = line_length * np.cos(np.deg2rad(angle1))
        y1 = line_length * np.sin(np.deg2rad(angle1))
        x2 = line_length * np.cos(np.deg2rad(angle2))
        y2 = line_length * np.sin(np.deg2rad(angle2))

        arc1 = points1[points1[:, 2]==1.0] * 1
        arc2 = points2[points2[:, 2]==1.0] * 1.2
        
        intersect_arc = np.copy(points1)
        intersect_arc[:, 2] = intersect_arc[:, 2] * points2[:, 2]
        intersect_arc = intersect_arc[intersect_arc[:, 2] == 1.]

        if not ax is None: 
            
            ax.scatter(arc1[:, 0], arc1[:, 1], c = arc1[:, 2])
            ax.scatter(arc2[:, 0], arc2[:, 1], c = arc2[:, 2]) 
            # ax.scatter(intersect_arc[:, 0], intersect_arc[:, 1])

            for i in range(10):
                ax.plot([0, w[i, 0]], [0, w[i, 1]])
                ax.annotate(f'{i}: {angle_arr[i]:.2f} | size: {np.linalg.norm(w[i, :]):.3f}', [w[i, 0], w[i, 1]])
        
            ax.axis('square')

        return x1, y1, x2, y2

    locs = []
    for i in range(10):
        locs.append(temp(i, line_length=line_length))
    
    return locs

class model_wrapper(nn.Module):
    def __init__(self, model): 
        super(model_wrapper, self).__init__()
        self.model = model 
    
    def forward(self, x):
        _, outs = model(x)
        return outs
    
class Lenetspp(nn.Module):
    def generate_matrix(deviation_factor=0.5):
        n_vectors = 10
        vectors = []
        base_angle = 2 * torch.pi / n_vectors

        angle = torch.rand(1) * 2 * torch.pi
        vector = torch.tensor([torch.cos(angle), torch.sin(angle)])
        vectors.append(vector)

        for i in range(1, n_vectors):
            deviation = (torch.rand(1) - 0.5) * deviation_factor
            angle += base_angle + deviation
            vector = torch.tensor([torch.cos(angle), torch.sin(angle)])
            vectors.append(vector)

        matrix = torch.stack(vectors)
        norms = torch.norm(matrix, dim=1, keepdim=True)
        matrix = matrix / norms * (1 + (torch.rand(matrix.size(0), 1) - 0.5) * 0.1)

        return matrix

    def __init__(self) -> None:
        super(Lenetspp, self).__init__()
        # assert feat_dim == 2, 'lenet++ has feat dim = 2'
        self.feat_dim = 2
        self.num_classes = 10
        # figure size + 1 - kernel size

        self.conv1x = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2x = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3x = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 128, 3, 3

        self.fc1 = nn.Linear(128 * 3 * 3, self.feat_dim)
        self.bn1 = nn.BatchNorm1d(self.feat_dim)
        self.prelu_1 = nn.PReLU()

        matrix = Lenetspp.generate_matrix(deviation_factor=0.5)

        # self.symmetrical = SymmetricalLayer(self.feat_dim, 10)
        self.softmax_weights = nn.Linear(self.feat_dim, 10, bias=False)
        with torch.no_grad():
            self.softmax_weights.weight.data = (matrix)

        self.grad_layer = [self.pool3, self.pool2, self.pool1]

        # Added for testing logit margin
        # nn.init.xavier_normal_(self.softmax_weights.weight)

    def normalize_weights(self, ): 
        with torch.no_grad():
            self.softmax_weights.weight.data = 1. * F.normalize(self.softmax_weights.weight.data, dim=1)

    def get_softmax_weights(self):
        return self.softmax_weights.weight.data

    def get_penultimate_layer(self, ):
        return self.softmax_weights

    def forward(self, x):
        x = self.conv1x(x)
        x = self.pool1(x)
        x = self.conv2x(x)
        x = self.pool2(x)
        x = self.conv3x(x)
        x = self.pool3(x)
        x = x.view(-1, 128 * 3 * 3)

        feats = self.bn1(self.fc1(x))
        feats = self.prelu_1(feats)

        #### feats = F.normalize(feats, p=2, dim=1)
        
        # feats, logits = self.symmetrical(feats)
        logits = self.softmax_weights(feats)
        return feats, logits

# def geometric_analysis(
#         features: np.ndarray, 
#         adv_features: np.ndarray, 
#         labels: np.ndarray,
#         preds: np.ndarray,
#         adv_preds: np.ndarray,
#         weights: np.ndarray,
#         name_to_save: str, 
#         path_to_save: str,
#         title: str = None,
#         zoom: bool = False
#     ):
#     output = {}


#     # Create a 2x2 subplot grid with custom widths for columns
#     fig = plt.figure(figsize=[10, 10], dpi=200)
#     gs = gridspec.GridSpec(1, 1, width_ratios=[0.99])
#     """
#     [0, 0] | [0, 1] | [0, 2]
#     [1, 0] | [1, 1] | [1, 2]
#     """

#     # Create a subplot that spans two rows (belongs to the first column)
#     ax3 = plt.subplot(gs[0,0])

#     c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
#         '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']    
    
#     # plot features, mean, weights, based on True label
#     if features.shape[0] < 2000 or zoom: marker_size = 4
#     elif features.shape[0] < 11000: marker_size = 3
#     else: marker_size = 1

#     marker_size = 15
    
#     # get boundaries
#     feat_dim = features.shape[1]

#     max_norm_f = np.quantile(np.linalg.norm(features, axis=1), q=0.95)    
#     locs = get_boundaries(weights, max_norm_f)

#     ############################################################################################
#     curr_ax = ax3
#     mean_classes_plot = []
#     max_norm = -1.0     
#     w_max_norm = -1.0
#     for class_num in range(len(np.unique(labels))):
#         feats_c = features[labels==class_num] # [k, 2]
#         mean_c = np.mean(feats_c, axis=0) # [2, ]
#         mean_classes_plot.append(mean_c)
#         max_norm = max(max_norm, np.linalg.norm(mean_c))
#         w_max_norm = max(w_max_norm, np.linalg.norm(weights[class_num, :]))

#     adv_mean_classes_plot = []
#     for class_num in range(len(np.unique(labels))):
#         adv_feats_c = adv_features[labels==class_num] # [k, 2]
#         adv_mean_c = np.mean(adv_feats_c, axis=0) # [2, ]
#         adv_mean_classes_plot.append(adv_mean_c)

#     for class_num in range(len(np.unique(labels))):
#         feats_c = features[labels==class_num] 
#         adv_feats_c = adv_features[labels==class_num]

#         # Plot class means
#         # curr_ax.annotate(f'{class_num}', [mean_classes_plot[class_num][0], mean_classes_plot[class_num][1]])
        
#         curr_ax.plot(feats_c[:, 0], 
#                     feats_c[:, 1],
#                     '.', c=c[class_num], alpha=0.85, 
#                     ms=marker_size, 
#                     label=class_num)
        
#         curr_ax.plot(adv_feats_c[:, 0], 
#                     adv_feats_c[:, 1],
#                     'x', c=c[class_num], alpha=0.85, 
#                     ms=6, 
#                     label=class_num)

#         # plot (resized) weights
#         resized_weight = (weights[class_num, :]/w_max_norm) * (max_norm)
#         curr_ax.plot([0, resized_weight[0]], [0, resized_weight[1]],
#                     c='black', lw=2, label=class_num)
#         curr_ax.plot([0, resized_weight[0]], [0, resized_weight[1]],
#                     c=c[class_num], lw=1.5, label=class_num)
#         curr_ax.annotate(f'{class_num}', [resized_weight[0], resized_weight[1]])
        
#         # plot boundaries
#         x1, y1, x2, y2 = locs[class_num]
#         plt.plot([0, x1], [0, y1], [0, x2], [0, y2], c=c[class_num], label=f'Both Angles', alpha=0.4, lw=1.25)
#         plt.fill_between([0, x1, x2, 0], [0, y1, y2, 0], color=c[class_num], alpha=0.1)
        
#     mean_classes_plot = np.vstack(mean_classes_plot)
#     curr_ax.set_title(name_to_save)
#     # ax.ylim(-200, 200)
#     # ax.xlim(-200, 200)
#     curr_ax.axis('square')
#     if zoom: 
#         curr_ax.set_xlim(2*min(mean_classes_plot[1, 0], mean_classes_plot[4, 0]), 2*max(mean_classes_plot[1, 0], mean_classes_plot[4, 0]))
#         curr_ax.set_ylim(-2, 2*max(mean_classes_plot_classes[1, 1], mean_classes_plot[4, 1]))
#     ############################################################################################

#     plt.title(title)
        
#     plt.tight_layout()
#     make_dir(path_to_save)
#     plt.suptitle(f'{feat_dim}-D features')
#     plt.savefig(f'{path_to_save}/{name_to_save}.jpg')
#     plt.clf()

#     return output


def geometric_analysis(
        features: np.ndarray, 
        adv_features: np.ndarray, 
        labels: np.ndarray,
        preds: np.ndarray,
        adv_preds: np.ndarray,
        weights: np.ndarray,
        name_to_save: str, 
        path_to_save: str,
        title: str = None,
        zoom: bool = False
    ):
    output = {}


    # Create a 2x2 subplot grid with custom widths for columns
    """
    [0, 0] | [0, 1] | [0, 2]
    [1, 0] | [1, 1] | [1, 2]
    """

    # Create a subplot that spans two rows (belongs to the first column)
    c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
        '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']    

    marker_size = 50
    
    # get boundaries
    feat_dim = features.shape[1]

    max_norm_f = np.quantile(np.linalg.norm(features, axis=1), q=0.95)    
    locs = get_boundaries(weights, max_norm_f)

    ############################################################################################
    mean_classes_plot = []
    max_norm = -1.0     
    w_max_norm = -1.0
    for class_num in range(len(np.unique(labels))):
        feats_c = features[labels==class_num] # [k, 2]
        mean_c = np.mean(feats_c, axis=0) # [2, ]
        mean_classes_plot.append(mean_c)
        max_norm = max(max_norm, np.linalg.norm(mean_c))
        w_max_norm = max(w_max_norm, np.linalg.norm(weights[class_num, :]))

    adv_mean_classes_plot = []
    for class_num in range(len(np.unique(labels))):
        adv_feats_c = adv_features[labels==class_num] # [k, 2]
        adv_mean_c = np.mean(adv_feats_c, axis=0) # [2, ]
        adv_mean_classes_plot.append(adv_mean_c)

    for class_num in range(len(np.unique(labels))):
        fig = plt.figure(figsize=[10, 10], dpi=200)

        feats_c = features[labels==class_num] 
        adv_feats_c = adv_features[labels==class_num]

        plt.scatter(feats_c[:, 0], 
            feats_c[:, 1],
            c=c[class_num],  # Face color
            edgecolors='black',  # Outline color
            alpha=0.85, 
            marker='o',  # Filled marker
            s=marker_size,  # Marker size (area)
            linewidths=1,
            label=class_num)
        
        plt.scatter(adv_feats_c[:, 0], 
            adv_feats_c[:, 1],
            c=c[class_num],  # Face color
            edgecolors='black',  # Outline color
            alpha=0.85, 
            marker='s',  # Filled marker
            s=marker_size,  # Marker size (area)
            linewidths=1,
            label=class_num)

        for inner_class_num in range(len(np.unique(labels))):
            # Plot class means
            plt.annotate(f'{inner_class_num}', [mean_classes_plot[inner_class_num][0], mean_classes_plot[inner_class_num][1]])
            
            # plot (resized) weights
            resized_weight = (weights[inner_class_num, :]/w_max_norm) * (max_norm)
            plt.plot([0, resized_weight[0]], [0, resized_weight[1]],
                        c='black', lw=2, label=inner_class_num)
            plt.plot([0, resized_weight[0]], [0, resized_weight[1]],
                        c=c[inner_class_num], lw=1.5, label=inner_class_num)
            plt.annotate(f'{inner_class_num}', [resized_weight[0], resized_weight[1]])
            
            # plot boundaries
            x1, y1, x2, y2 = locs[inner_class_num]
            plt.plot([0, x1], [0, y1], [0, x2], [0, y2], c=c[inner_class_num], label=f'Both Angles', alpha=0.4, lw=1.25)
            plt.fill_between([0, x1, x2, 0], [0, y1, y2, 0], color=c[inner_class_num], alpha=0.1)

        plt.axis('square')        
        plt.title(title)

        plt.tight_layout()
        make_dir(path_to_save)
        plt.suptitle(f'class {class_num} - {feat_dim}-D features')
        plt.savefig(f'{path_to_save}/{class_num}-{name_to_save}.jpg')
        plt.clf()
        
    return output

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
DATASET = 'MNIST'

method = 'CE'
loss_fn = nn.CrossEntropyLoss()
model = Lenetspp().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30 
BATCH_SIZE = 128

TREE_ROOT = ''
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

from torch.utils.data import DataLoader, Subset, Dataset
random_indices = torch.randperm(len(testset))[:256]
sampled_dataset = Subset(testset, random_indices)
sampled_dataloader = DataLoader(sampled_dataset, batch_size=256, shuffle=False)
sampled_data, sampled_labels = next(iter(sampled_dataloader))
sampled_data = sampled_data.to(device)
sampled_labels = sampled_labels.to(device)

from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
adv_model = model_wrapper(model, )

adversary = LinfPGDAttack(
    adv_model,
    loss_fn=loss_fn,
    eps=0.3,
    nb_iter=40,
    eps_iter=0.01,
    rand_init=False,
    clip_min=0.0,
    clip_max=1.0,
    targeted=False,
) 

sampled_adversary = LinfPGDAttack(
    adv_model,
    loss_fn=loss_fn,
    eps=0.3,
    nb_iter=40,
    eps_iter=0.01,
    rand_init=True,
    clip_min=0.0,
    clip_max=1.0,
    targeted=False,
)  

from tqdm import trange 
from sklearn.metrics import accuracy_score
for epoch in trange(epochs): 
    all_preds = []
    all_labels = []
    for inputs, labels in trainloader: 
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        adv_model.eval()
        with ctx_noparamgrad_and_eval(adv_model):
            inputs = adversary.perturb(inputs, labels)
        
        model.train()
        feats, outs = model(inputs)
        loss = loss_fn(outs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_preds.append(torch.argmax(outs, dim=1).cpu().detach().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print(f'{epoch} acc: ', accuracy_score(all_labels, all_preds))

    adv_model.eval()
    with ctx_noparamgrad_and_eval(adv_model):
        adv_inputs = sampled_adversary.perturb(sampled_data, sampled_labels)
    model.eval()
    adv_feats, adv_outs = model(adv_inputs)
    adv_preds = torch.argmax(adv_outs, dim=1)
    cln_feats, cln_outs = model(sampled_data)
    cln_preds = torch.argmax(cln_outs, dim=1)
    
    print('adv acc', accuracy_score(sampled_labels.cpu().numpy(), adv_preds.cpu().numpy()))
    print('cln acc', accuracy_score(sampled_labels.cpu().numpy(), cln_preds.cpu().numpy()))

    geometric_analysis(
        features=cln_feats.cpu().detach().numpy(), 
        adv_features=adv_feats.cpu().detach().numpy(), 
        labels=sampled_labels.detach().cpu().numpy(), 
        preds=cln_preds.detach().cpu().numpy(), 
        adv_preds=adv_preds.detach().cpu().numpy(), 
        weights=model.get_softmax_weights().detach().cpu().numpy(), 
        name_to_save=f'test{epoch}', path_to_save='temp', title=f'{epoch}')

