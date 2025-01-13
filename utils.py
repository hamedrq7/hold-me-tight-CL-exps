import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
import torchvision
import torchvision.transforms as transforms
import torch_dct
import numpy as np
import copy
from typing import Dict, List
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import collections
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

import os 
def make_dir(path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)


from torch.autograd import Variable

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        # print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    _, f_image = net(Variable(image[None, :, :, :], requires_grad=True))
    f_image = f_image.data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    _, fs = net(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        _, fs = net(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image

class CenterLoss(nn.Module):
    """
    https://github.com/jxgu1016/MNIST_center_loss_pytorch
    """
    def __init__(self, num_classes, feat_dim, size_average=True):
        """
        Parameters
        ----------
            num_classes: int
                number of classes
            feat_dim: int
                feature's dimension
            size_average: bool
                author mentioned use of this parameter in github page. no use for us.
        """
        super().__init__()
        self.num_classes = num_classes 
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # self.centers = nn.Parameter(torch.zeros(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.size_average = size_average

        self.num_classes = num_classes

    def forward(self, features, labels, grad=True) -> Dict[str, torch.Tensor]:
        """
        m = batch_size
        label - [m, ]
        feat - [m, dim] - in case of mnist toy example: [m, 2]
        """
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        # To check the dim of centers and features
        if features.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(
                    self.feat_dim, features.size(1)
                )
            )

        batch_size_tensor = features.new_empty(1).fill_(
            batch_size if self.size_average else 1
        )

        if grad:
            loss = self.centerlossfunc(
                features, labels, self.centers, batch_size_tensor
            )
        else:
            # Stops gradients for centers...
            loss = self.centerlossfunc(
                features, labels, self.centers.clone().detach(), batch_size_tensor
            )
        return loss

from torch.autograd.function import Function

class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())  # [m, feat_dim]
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())  # [m, feat_dim]

        diff = centers_batch - feature  # c_yi - x_i
        # init every iteration
        counts = centers.new_ones(centers.size(0))  # [c, ] of ones
        ones = centers.new_ones(label.size(0))  # [m, ] of ones
        grad_centers = centers.new_zeros(centers.size())  # [c, feat_dim] of zeros

        # 1-D case:
        # grad_centers[label[i]] += ones[i]
        counts = counts.scatter_add_(0, label.long(), ones)

        grad_centers.scatter_add_(
            0, label.unsqueeze(1).expand(feature.size()).long(), diff
        )  # [c, feat_dim]

        grad_centers = grad_centers / counts.view(-1, 1)

        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None


def train_cl(model, device, trans, trainloader, testloader, epochs, opt, loss_fun, 
             lr_schedule, save_train_dir, center_lr = 0.5, alpha = 0.1):
    cl = CenterLoss(10, model.fd).to(device)
    optimizer4center = torch.optim.SGD(cl.parameters(), lr=center_lr)
    
    # lr_schedule = lambda t: np.interp([t], [0, epochs * 2 // 5, epochs], [0, max_lr, 0])[0]
    # loss_fun = nn.CrossEntropyLoss()

    print('Starting training...')
    print()

    for epoch in range(epochs):
        print('Epoch', epoch)
        train_loss_sum = 0
        train_cl_loss_sum = 0
        train_acc_sum = 0
        train_n = 0

        model.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            opt.param_groups[0].update(lr=lr)

            feats, output = model(trans(inputs))
            cl_loss = cl(feats, targets)
            loss = loss_fun(output, targets) + float(alpha) * cl_loss

            opt.zero_grad()
            optimizer4center.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer4center.step()
            opt.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_cl_loss_sum += cl_loss.item() * targets.size(0)
            train_acc_sum += (output.max(1)[1] == targets).sum().item()
            train_n += targets.size(0)

            if batch_idx % 100 == 0:
                print('Batch idx: %d(%d)\tTrain Acc: %.3f%%\tTrain Loss: %.3f' %
                      (batch_idx, epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

        print('\nTrain Summary\tEpoch: %d | Train Acc: %.3f%% | Train Loss: %.3f | Train CLLoss: %.3f' %
              (epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n, train_cl_loss_sum/train_n))

        test_acc, test_loss = test(model, trans, testloader)
        print('Test  Summary\tEpoch: %d | Test Acc: %.3f%% | Test Loss: %.3f\n' % (epoch, test_acc, test_loss))

    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()

    torch.save(state_dict, save_train_dir + 'model.t7')

    return model

def train(model, trans, trainloader, testloader, epochs, opt, loss_fun, lr_schedule, save_train_dir):

    # lr_schedule = lambda t: np.interp([t], [0, epochs * 2 // 5, epochs], [0, max_lr, 0])[0]
    # loss_fun = nn.CrossEntropyLoss()

    print('Starting training...')
    print()

    for epoch in range(epochs):
        print('Epoch', epoch)
        train_loss_sum = 0
        train_acc_sum = 0
        train_n = 0

        model.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            opt.param_groups[0].update(lr=lr)

            feats, output = model(trans(inputs))
            loss = loss_fun(output, targets)

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_acc_sum += (output.max(1)[1] == targets).sum().item()
            train_n += targets.size(0)

            if batch_idx % 100 == 0:
                print('Batch idx: %d(%d)\tTrain Acc: %.3f%%\tTrain Loss: %.3f' %
                      (batch_idx, epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

        print('\nTrain Summary\tEpoch: %d | Train Acc: %.3f%% | Train Loss: %.3f' %
              (epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

        test_acc, test_loss = test(model, trans, testloader)
        print('Test  Summary\tEpoch: %d | Test Acc: %.3f%% | Test Loss: %.3f\n' % (epoch, test_acc, test_loss))

    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()

    torch.save(state_dict, save_train_dir + 'model.t7')

    return model


def test(model, trans, testloader):
    loss_fun = nn.CrossEntropyLoss()
    test_loss_sum = 0
    test_acc_sum = 0
    test_n = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            feats, output = model(trans(inputs))
            loss = loss_fun(output, targets)

            test_loss_sum += loss.item() * targets.size(0)
            test_acc_sum += (output.max(1)[1] == targets).sum().item()
            test_n += targets.size(0)

        test_loss = (test_loss_sum / test_n)
        test_acc = (100 * test_acc_sum / test_n)

        return test_acc, test_loss


def subspace_deepfool(im, model, trans, num_classes=10, overshoot=0.02, max_iter=100, Sp=None, device=DEVICE):
    image = copy.deepcopy(im)
    input_shape = image.size()

    feats, f_image = model(trans(Variable(image, requires_grad=True)))
    f_image = f_image.view((-1,))
    I = f_image.argsort(descending=True)
    I = I[0:num_classes]
    label_orig = I[0]

    pert_image = copy.deepcopy(image)

    r = torch.zeros(input_shape).to(device)

    label_pert = label_orig
    loop_i = 0

    while label_pert == label_orig and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        feats, fs = model(trans(x))

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            if Sp is None:
                pert_k = torch.abs(f_k) / w_k.norm()
            else:
                pert_k = torch.abs(f_k) / torch.matmul(Sp.t(), w_k.view([-1, 1])).norm()

            if pert_k < pert:
                pert = pert_k + 0.
                w = w_k + 0.

        if Sp is not None:
            w = torch.matmul(Sp, torch.matmul(Sp.t(), w.view([-1, 1]))).reshape(w.shape)

        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()
        r = r + r_i

        pert_image = pert_image + r_i

        feats, ooout = model(trans(Variable(image + (1 + overshoot) * r, requires_grad=False)))
        label_pert = torch.argmax(ooout.data).item()

        loop_i += 1

    return (1 + overshoot) * r, loop_i, label_orig, label_pert, image + (1 + overshoot) * r


def compute_margin_distribution(model, trans, dataloader, subspace_list, path, proc_fun=None):
    margins = []

    print('Measuring margin distribution...')
    for s, Sp in enumerate(subspace_list):
        Sp = Sp.to(DEVICE)
        sp_margin = []

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            if proc_fun:
                inputs = proc_fun(inputs)

            adv_perts = torch.zeros_like(inputs)
            for n, im in enumerate(inputs):
                adv_perts[n], _, _, _, _ = subspace_deepfool(im, model, trans, Sp=Sp)

            sp_margin.append(adv_perts.cpu().view([-1, np.prod(inputs.shape[1:])]).norm(dim=[1]))
        
        sp_margin = torch.cat(sp_margin)
        margins.append(sp_margin.numpy())
        print('Subspace %d:\tMedian margin: %5.5f' % (s, np.median(sp_margin)))

    np.save(path, margins)
    return np.array(margins)


def kron(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def generate_subspace_list(subspace_dim, dim, subspace_step, channels):
    subspace_list = []
    idx_i = 0
    idx_j = 0
    while (idx_i + subspace_dim - 1 <= dim - 1) and (idx_j + subspace_dim - 1 <= dim - 1):

        S = torch.zeros((subspace_dim, subspace_dim, dim, dim), dtype=torch.float32).to(DEVICE)
        for i in range(subspace_dim):
            for j in range(subspace_dim):
                dirac = torch.zeros((dim, dim), dtype=torch.float32, device=DEVICE)
                dirac[idx_i + i, idx_j + j] = 1.
                S[i, j] = torch_dct.idct_2d(dirac, norm='ortho')

        Sp = S.view(subspace_dim * subspace_dim, dim * dim)
        if channels > 1:
            Sp = kron(torch.eye(channels, dtype=torch.float32, device=DEVICE), Sp)

        Sp = Sp.t()

        Sp = Sp.to('cpu')
        subspace_list.append(Sp)

        idx_i += subspace_step
        idx_j += subspace_step

    return subspace_list

def get_mnist_eval(testset, num_samples=100, batch_size=128, seed=111): 
        import random
        def seed_worker(worker_id):
            # worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(seed)
            random.seed(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        indices = np.random.choice(len(testset), num_samples, replace=False)

        eval_dataset = torch.utils.data.Subset(testset, indices[:num_samples])
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size,
                                                  generator=g,
                                                  worker_init_fn = seed_worker,
                                                shuffle=False, num_workers=2, pin_memory=True if DEVICE == 'cuda' else False)
        
        return eval_dataset, eval_loader, num_samples

import matplotlib.pyplot as plt 

def plot_norms(norms, what_norm, title, path_to_save): 
    # Plot L2 norm histogram
    plt.figure(figsize=(8, 6))
    plt.hist(norms, bins=20, edgecolor='k')
    plt.title(f"{title}") # Histogram of $L_2$ Norms of Minimal Perturbations
    plt.xlabel(f"{what_norm}")
    plt.ylabel("Frequency")
    plt.savefig(f"{path_to_save}/{what_norm}_norm_histogram.png")

def get_dataset_loaders(dataset, dataset_dir, batch_size=128, seed=111):
    import random
    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(seed)
        random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    pin_memory = True if DEVICE == 'cuda' else False

    if dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root=dataset_dir['train'], download=True, train=True, transform=torchvision.transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=dataset_dir['val'], download=True, train=False, transform=torchvision.transforms.ToTensor())

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=pin_memory, generator=g, worker_init_fn=seed_worker)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=pin_memory, generator=g, worker_init_fn=seed_worker)

        mean = torch.tensor([0.1307], device=DEVICE)[None, :, None, None]
        std = torch.tensor([0.3081], device=DEVICE)[None, :, None, None]

    elif dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir['train'], download=True, train=True, transform=torchvision.transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=dataset_dir['val'], download=True, train=False, transform=torchvision.transforms.ToTensor())

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=pin_memory, generator=g, worker_init_fn=seed_worker)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=pin_memory, generator=g, worker_init_fn=seed_worker)

        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE)[None, :, None, None]
        std = torch.tensor([0.247, 0.243, 0.261], device=DEVICE)[None, :, None, None]

    elif dataset == 'ImageNet':

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.ImageFolder(root=dataset_dir['train'], transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=dataset_dir['val'], transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=True)

        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float, device=DEVICE)[None, :, None, None]
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float, device=DEVICE)[None, :, None, None]

    else:
        raise NotImplementedError

    return trainloader, testloader, trainset, testset, mean, std


def get_processed_dataset_loaders(proc_fun, dataset, dataset_dir, batch_size=128):

    pin_memory = True if DEVICE == 'cuda' else False

    if dataset == 'MNIST':
        orig_trainset = torchvision.datasets.MNIST(root=dataset_dir['train'], download=True, train=True, transform=torchvision.transforms.ToTensor())
        orig_testset = torchvision.datasets.MNIST(root=dataset_dir['val'], download=True, train=False, transform=torchvision.transforms.ToTensor())

        # trainset = torch.utils.data.TensorDataset(proc_fun(torch.tensor(trainset.data).type(torch.float32).permute([-1, 1, 28, 28]) / 255.), torch.tensor(trainset.targets))
        # testset = torch.utils.data.TensorDataset(proc_fun(torch.tensor(testset.data).type(torch.float32).permute([-1, 1, 28, 28]) / 255.), torch.tensor(testset.targets))

        trainset = torch.utils.data.TensorDataset(proc_fun(orig_trainset.data.type(torch.float32).unsqueeze(1) / 255.), orig_trainset.targets)
        testset = torch.utils.data.TensorDataset(proc_fun(orig_testset.data.type(torch.float32).unsqueeze(1) / 255.), orig_testset.targets)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=pin_memory)

        mean = torch.tensor([0.1307], device=DEVICE)[None, :, None, None]
        std = torch.tensor([0.3081], device=DEVICE)[None, :, None, None]

        proc_mean = torch.as_tensor(trainset.tensors[0].mean(axis=(0, 2, 3)), dtype=torch.float, device=DEVICE)[None, :, None, None]
        proc_std = torch.as_tensor(trainset.tensors[0].std(axis=(0, 2, 3)), dtype=torch.float, device=DEVICE)[None, :, None, None]

    elif dataset == 'CIFAR10':
        orig_trainset = torchvision.datasets.CIFAR10(root=dataset_dir['train'], download=True, train=True, transform=torchvision.transforms.ToTensor())
        orig_testset = torchvision.datasets.CIFAR10(root=dataset_dir['val'], download=True, train=False, transform=torchvision.transforms.ToTensor())

        trainset = torch.utils.data.TensorDataset(proc_fun(torch.tensor(orig_trainset.data).type(torch.float32).permute([0, 3, 1, 2]) / 255.), torch.tensor(orig_trainset.targets))
        testset = torch.utils.data.TensorDataset(proc_fun(torch.tensor(orig_testset.data).type(torch.float32).permute([0, 3, 1, 2]) / 255.), torch.tensor(orig_testset.targets))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=pin_memory)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=pin_memory)

        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE)[None, :, None, None]
        std = torch.tensor([0.247, 0.243, 0.261], device=DEVICE)[None, :, None, None]

        proc_mean = torch.as_tensor(trainset.tensors[0].mean(axis=(0, 2, 3)), dtype=torch.float, device=DEVICE)[None, :, None, None]
        proc_std = torch.as_tensor(trainset.tensors[0].std(axis=(0, 2, 3)), dtype=torch.float, device=DEVICE)[None, :, None, None]

    else:
        raise NotImplementedError

    return trainloader, testloader, trainset, testset, mean, std, proc_mean, proc_std

