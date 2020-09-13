# PyTorch-accelerated implementation of sampling/discriminator
# based implementation of mutual information and conditional mutual
# information following Mukherjee, Asnani,
# and Kannan (2019; https://arxiv.org/abs/1906.01824)
#
# Also: Implementation of RKHS-based complexity-regularisation, to improve 
# reliability and stability of KL-estimates, following Ghimire, Gyawali, 
# and Wang (2020; https://arxiv.org/abs/2002.11187)
# 
# For a nice introduction on sampling based approximation of KL-Divergences
# see Mescheder, Nowozin, and Geiger (2017; https://arxiv.org/abs/1701.04722)

# To select, which gpu to use, you can use the following lines in your main code:
#
# import torch
#
# ...
#
# torch.cuda.set_device(which_gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

# Discriminator, whose last layer is sampled from a Gaussian distribution, as described in
# Ghimire et al. (https://arxiv.org/abs/2002.11187)
class discriminator_rkhs(nn.Module):

    def __init__(self, d=128, n_hidden = 128, use_cuda = False):
        super(discriminator_rkhs, self).__init__()
        
        self.d = d
        self.n_hidden = n_hidden
        
        self.fc1 = nn.Linear(d, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        
        self.use_cuda = use_cuda
        
        # Randomized last layer, according to: https://arxiv.org/pdf/2002.11187.pdf
        self.theta = torch.nn.Parameter(torch.empty((1, n_hidden),requires_grad = True))
        torch.nn.init.normal_(self.theta, mean=0.0, std=1.0)
                                 
        self.L = torch.nn.Parameter(torch.empty((n_hidden, n_hidden), requires_grad = True))
        torch.nn.init.orthogonal_(self.L, gain=1.41) 
        
    def weight_init(self):
        
        for m in self._modules:
            #print(self._modules[m])
            m = self._modules[m]
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,1.41)
                m.bias.data.fill_(0.0)
    
    def forward(self, x):
        
        n_samples = x.shape[0]
        
        x = F.relu((self.fc1(x)))
        phi = F.relu((self.fc2(x)))
        
        noise = torch.normal(0.0, 1.0, (n_samples, self.n_hidden))
        
        if self.use_cuda:
            noise = noise.cuda()
        
        cov = torch.tensordot(noise, self.L, ([1],[1]))
        
        w = self.theta + cov
        
        result = torch.bmm(phi.unsqueeze(dim=1),w.unsqueeze(dim=2)).squeeze()
                            
        wwT = torch.matmul(self.theta.transpose(0,1),self.theta)
        
        SIGMA = torch.matmul(self.L, self.L.transpose(0,1))
        
        multiplied = torch.tensordot(phi, wwT + SIGMA, ([1],[1]))
        
        return result.unsqueeze(dim=1), phi, multiplied 
    

# Standard discriminator, consisting of multi-layer perceptron with 2 hidden layers    
class discriminator(nn.Module):

    def __init__(self, d=128, n_hidden = 100):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(d, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)
        
    def weight_init(self):
        
        for m in self._modules:
            m = self._modules[m]
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight,1.41)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
    
        return x

# Discriminator based approximation of mutual information MI(X,Y)
class mutual_information:
    def __init__(self, dimX, dimY, data, clip_grad = False, clip_norm = 100, use_cuda = True, learning_rate = 1e-4, n_hidden = 100, smooth = 0.0, log_mi_every = 10, log_mi_samples = 10000):
        self.dimX = dimX
        self.dimY = dimY
        self.data = data
        self.n = self.data.shape[0]
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.discriminator = discriminator(d=dimX + dimY, n_hidden = n_hidden)
        self.smooth = smooth
        self.log_mi_every = log_mi_every
        self.log_mi_samples = log_mi_samples
        
        if use_cuda:
            self.discriminator = self.discriminator.cuda()
            
        self.discriminator.weight_init()
            
        self.optimizer = optim.RMSprop(self.discriminator.parameters(), lr=learning_rate)
        
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm
        
        self.use_cuda = use_cuda
        
        self.losses = []
        
        self.approximations = []
        
    def train(self, n_iter = 1000, batch_size = 1000):
        
        label_joint = torch.ones((batch_size,1))
        label_prod = torch.zeros((batch_size,1))
        
        if self.use_cuda:
            label_joint = label_joint.cuda()
            label_prod = label_prod.cuda()
        
        for i in tqdm(range(n_iter)):
            
            items = np.random.choice(self.n,batch_size)
            
            batch_joint = torch.from_numpy( self.data[items,:] ).float()
            
            items_x = np.random.choice(self.n,batch_size)
            items_y = np.random.choice(self.n,batch_size)
            
            batch_prod = torch.from_numpy(
                np.concatenate( (self.data[items_x,:self.dimX], 
                                 self.data[items_y,self.dimX:]),
                                        axis = 1 ) ).float()
            
            if self.use_cuda:
                batch_joint = batch_joint.cuda()
                batch_prod = batch_prod.cuda()
            
            # CAVE: INSTANCE NOISE WILL *ALTER* THE DISTRIBUTIONS, WHOSE
            # MUTUAL INFORMATION YOU ARE CALCULATING. BE AWARE OF THIS!
            if self.smooth > 1e-12:
                print('SMOOTHING')
                batch_joint += self.smooth*torch.random.randn(batch_joint.shape)
                batch_prod += self.smooth*torch.random.randn(batch_prod.shape)
            
            x_joint = self.discriminator(batch_joint)
            x_prod = self.discriminator(batch_prod)
            
            loss = self.BCE_loss(x_joint, label_joint) + self.BCE_loss(x_prod, label_prod)
            
            loss.backward()
        
            if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_norm)
        
            self.optimizer.step()
            
            self.losses.append(loss.cpu().item())
            
            if self.log_mi_every > 0 and i % self.log_mi_every == 0:
                self.approximations.append(self.approximate(self.log_mi_samples))


    def approximate(self, n_samples = 10000):
        samples_joint = torch.from_numpy( self.data[np.random.choice(self.n,n_samples),:] ).float()
        
        if self.use_cuda:
            samples_joint = samples_joint.cuda()
        
        mi = self.discriminator(samples_joint).mean().cpu().item()
            
        return mi
                
# Discriminator based approximation of mutual information MI(X,Y)
class mutual_information_rkhs:
    def __init__(self, dimX, dimY, data, clip_grad = False, clip_norm = 100, use_cuda = True, learning_rate = 1e-4, n_hidden = 100, penalty = 1e-3, smooth = 0.0, log_mi_every = 10, log_mi_samples = 10000):
        self.dimX = dimX
        self.dimY = dimY
        self.data = data
        self.n = self.data.shape[0]
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.discriminator = discriminator_rkhs(d=dimX + dimY, n_hidden = n_hidden, use_cuda = use_cuda)
        self.penalty = penalty
        self.smooth = smooth
        self.log_mi_every = log_mi_every
        self.log_mi_samples = log_mi_samples
        
        
        if use_cuda:
            self.discriminator = self.discriminator.cuda()
            
        self.discriminator.weight_init()
            
        self.optimizer = optim.RMSprop(self.discriminator.parameters(), lr=learning_rate)
        
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm
        
        self.use_cuda = use_cuda
        
        self.losses = []
        
        self.approximations = []
        
    def train(self, n_iter = 1000, batch_size = 1000):
        
        label_joint = torch.ones((batch_size,1))
        label_prod = torch.zeros((batch_size,1))
        
        if self.use_cuda:
            label_joint = label_joint.cuda()
            label_prod = label_prod.cuda()
        
        for i in tqdm(range(n_iter)):
            
            items = np.random.choice(self.n,batch_size)
            
            batch_joint = torch.from_numpy( self.data[items,:] ).float()
            
            items_x = np.random.choice(self.n,batch_size)
            items_y = np.random.choice(self.n,batch_size)
            
            batch_prod = torch.from_numpy(
                np.concatenate( (self.data[items_x,:self.dimX], 
                                 self.data[items_y,self.dimX:]),
                                        axis = 1 ) ).float()   
            
            if self.use_cuda:
                batch_joint = batch_joint.cuda()
                batch_prod = batch_prod.cuda()
                
            # CAVE: INSTANCE NOISE WILL *ALTER* THE DISTRIBUTIONS, WHOSE
            # MUTUAL INFORMATION YOU ARE CALCULATING. BE AWARE OF THIS!
            if self.smooth > 1e-12:
                print('SMOOTHING')
                batch_joint += self.smooth*torch.random.randn(batch_joint.shape)
                batch_prod += self.smooth*torch.random.randn(batch_prod.shape)
            
            x_joint, phi_joint, mult_joint = self.discriminator(batch_joint)
            x_prod, phi_prod, mult_prod = self.discriminator(batch_prod)
            
            phi = torch.cat((phi_joint, phi_prod), 0)
            mult = torch.cat((mult_joint, mult_prod), 0)
            
            kernel = torch.tensordot(phi, mult, ([1],[1]))
            
            S = kernel.max()
            
            loss = self.BCE_loss(x_joint, label_joint) + self.BCE_loss(x_prod, label_prod)
            
            loss = loss + self.penalty*S
            
            loss.backward()
        
            if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_norm)
        
            self.optimizer.step()
            
            self.losses.append(loss.cpu().item())
            
            if self.log_mi_every > 0 and i % self.log_mi_every == 0:
                self.approximations.append(self.approximate(self.log_mi_samples))


    def approximate(self, n_samples = 10000):
        samples_joint = torch.from_numpy( self.data[np.random.choice(self.n,n_samples),:] ).float()
        
        if self.use_cuda:
            samples_joint = samples_joint.cuda()
        
        mi = self.discriminator(samples_joint)[0].mean().cpu().item()
            
        return mi
    
# Discriminator based approximation of conditional mutual information MI(X,Y|Z)
class conditional_mutual_information:
    def __init__(self, dimX, dimY, dimZ, data, clip_grad = False, clip_norm = 100, use_cuda = True, both_dir = True, learning_rate = 1e-4, n_hidden = 100, use_rkhs = False, rkhs_penalty = 0.0, log_mi_every = 10, log_mi_samples = 10000):
        self.dimX = dimX
        self.dimY = dimY
        self.dimZ = dimZ
        self.data = data
        self.n = self.data.shape[0]
        
        self.clip_grad = clip_grad
        self.clip_norm = clip_norm
        
        self.use_cuda = use_cuda
        
        self.both_dir = both_dir
        
        self.dataX = data[:,:dimX]
        self.dataY = data[:,dimX:(dimX+dimY)]
        self.dataZ = data[:,-dimZ:]
        
        dataXZ = np.concatenate( (self.dataX, self.dataZ), 1 )
        
        if use_rkhs:
            self.MI_X_YZ = mutual_information_rkhs(
                dimX, dimY + dimZ, data, penalty = rkhs_penalty, clip_grad = clip_grad, clip_norm = clip_norm, 
                use_cuda = use_cuda, learning_rate = learning_rate, n_hidden = n_hidden,
                log_mi_every = log_mi_every, log_mi_samples = log_mi_samples)
        
            self.MI_X_Z = mutual_information_rkhs(
                dimX, dimZ, dataXZ, penalty = rkhs_penalty, clip_grad = clip_grad, clip_norm = clip_norm, 
                use_cuda = use_cuda, learning_rate = learning_rate, n_hidden = n_hidden,
                log_mi_every = log_mi_every, log_mi_samples = log_mi_samples)
        else:
            self.MI_X_YZ = mutual_information(
                dimX, dimY + dimZ, data, clip_grad = clip_grad, clip_norm = clip_norm, 
                use_cuda = use_cuda, learning_rate = learning_rate, n_hidden = n_hidden,
                log_mi_every = log_mi_every, log_mi_samples = log_mi_samples)
        
            self.MI_X_Z = mutual_information(
                dimX, dimZ, dataXZ, clip_grad = clip_grad, clip_norm = clip_norm, 
                use_cuda = use_cuda, learning_rate = learning_rate, n_hidden = n_hidden,
                log_mi_every = log_mi_every, log_mi_samples = log_mi_samples)
        
        if both_dir:
            dataYXZ = np.concatenate( (self.dataY,self.dataX,self.dataZ), 1 )
            dataYZ = np.concatenate( (self.dataY, self.dataZ), 1 )
            
            if use_rkhs:
                self.MI_Y_XZ = mutual_information_rkhs(
                    dimY, dimX + dimZ, dataYXZ, penalty = rkhs_penalty, clip_grad = clip_grad, clip_norm = clip_norm, 
                    use_cuda = use_cuda, learning_rate = learning_rate, n_hidden = n_hidden,
                log_mi_every = log_mi_every, log_mi_samples = log_mi_samples)
        
                self.MI_Y_Z = mutual_information_rkhs(
                    dimY, dimZ, dataYZ, penalty = rkhs_penalty, clip_grad = clip_grad, clip_norm = clip_norm, 
                    use_cuda = use_cuda, learning_rate = learning_rate, n_hidden = n_hidden,
                log_mi_every = log_mi_every, log_mi_samples = log_mi_samples)
            else:
                self.MI_Y_XZ = mutual_information(
                    dimY, dimX + dimZ, dataYXZ, clip_grad = clip_grad, clip_norm = clip_norm, 
                    use_cuda = use_cuda, learning_rate = learning_rate, n_hidden = n_hidden,
                log_mi_every = log_mi_every, log_mi_samples = log_mi_samples)
        
                self.MI_Y_Z = mutual_information(
                    dimY, dimZ, dataYZ, clip_grad = clip_grad, clip_norm = clip_norm, 
                    use_cuda = use_cuda, learning_rate = learning_rate, n_hidden = n_hidden,
                log_mi_every = log_mi_every, log_mi_samples = log_mi_samples)
        
        
    def train(self, n_iter = 1000, batch_size = 1000, n_samples = 10000):
        
        print('training MI(X;YZ)...')
        self.MI_X_YZ.train(n_iter, batch_size)
        print('training MI(X;Z)...')
        self.MI_X_Z.train(n_iter, batch_size)
        
        if self.both_dir:
            print('training MI(Y;XZ)...')
            self.MI_Y_XZ.train(n_iter, batch_size)
            print('training MI(Y;Z)...')
            self.MI_Y_Z.train(n_iter, batch_size)
            

    def approximate(self, n_samples = 10000):
        
        mi_x_yz = self.MI_X_YZ.approximate(n_samples)
        mi_x_z = self.MI_X_Z.approximate(n_samples)
        
        cmi_f = mi_x_yz - mi_x_z
        
        if self.both_dir:
            mi_y_xz = self.MI_Y_XZ.approximate(n_samples)
            mi_y_z = self.MI_Y_Z.approximate(n_samples)
            
            cmi_b = mi_y_xz - mi_y_z
            
            return cmi_f, cmi_b
            
        return cmi_f