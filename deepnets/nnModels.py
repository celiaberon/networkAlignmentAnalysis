import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms

import tools.alignmentAnalysisTools as aat


class CNN2P2(nn.Module):

    """
    CNN with 2 convolutional layers, a max pooling stage, and 2 feedforward
    layers.
    Activation function is Relu by default (but can be chosen with
    convActivation and linearActivation).
    Output activation function is identity, because we're using
    CrossEntropyLoss.
    """

    def __init__(self, convActivation=F.relu, linearActivation=F.relu):
        super().__init__()
        self.numLayers = 4
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(256, 256)  # 4608, 512
        self.o = nn.Linear(256, 10)  # 512, 10

        self.convActivation = convActivation
        self.linearActivation = linearActivation

    def forward(self, x):
        self.c1 = self.convActivation(self.conv1(x))
        self.c2 = self.maxPool(self.convActivation(self.conv2(self.c1)))
        self.f1 = self.linearActivation(self.fc1(torch.flatten(self.c2, 1)))
        self.out = self.o(self.f1)
        return self.out

    def getDropout(self):
        return None

    def setDropout(self, dropout):
        return None

    def getActivations(self, x, usePrevious=False):
        if usePrevious:
            activations = [
                self.c1,
                self.c2,
                self.f1,
                self.out
            ]
        else:
            out = self.forward(x)
            activations = []
            activations.append(self.c1)
            activations.append(self.c2)
            activations.append(self.f1)
            activations.append(self.out)
        return activations

    def getNetworkWeights(self, onlyFF=False):
        netWeights = []
        if not onlyFF:
            netWeights.append(self.conv1.weight.data.clone().detach())
            netWeights.append(self.conv2.weight.data.clone().detach())
        netWeights.append(self.fc1.weight.data.clone().detach())
        netWeights.append(self.o.weight.data.clone().detach())
        return netWeights

    def compareNetworkWeights(self, initWeights):
        currWeights = self.getNetworkWeights()
        deltaWeights = []
        for iw, cw in zip(initWeights, currWeights):
            iw = torch.flatten(iw, 1)
            cw = torch.flatten(cw, 1)
            deltaWeights.append(torch.norm(cw - iw, dim=1))
        return deltaWeights

    def measureSimilarity(self, x, usePrevious=False):
        activations = self.getActivations(x, usePrevious=usePrevious)          
        similarity = []
        similarity.append(torch.mean(aat.similarityConvLayer(x, self.conv1),
                                     axis=1))
        similarity.append(torch.mean(aat.similarityConvLayer(activations[0],
                                                             self.conv2),
                                     axis=1))
        similarity.append(aat.similarityLinearLayer(torch.flatten(activations[1], 1),
                                                    self.fc1))
        similarity.append(aat.similarityLinearLayer(activations[2], self.o))
        return similarity

    def measureAlignment(self, x, eachLook=True, usePrevious=False):
        activations = self.getActivations(x, usePrevious=usePrevious) 
        alignment = []
        if eachLook:
            alignment.append(torch.mean(aat.alignmentConvLayer(x, self.conv1),
                                        axis=1))
            alignment.append(torch.mean(aat.alignmentConvLayer(activations[0],
                                                               self.conv2),
                                        axis=1))
        else:
            alignment.append(aat.alignmentConvLayer(x, self.conv1,
                                                    eachLook=eachLook))
            alignment.append(aat.alignmentConvLayer(activations[0], self.conv2,
                                                    eachLook=eachLook))
        alignment.append(aat.alignmentLinearLayer(torch.flatten(activations[1], 1),
                                                  self.fc1))
        alignment.append(aat.alignmentLinearLayer(activations[2], self.o))
        return alignment

    def manualShape(self, evals, evecs, DEVICE, evalTransform=None):
        if evalTransform is None:
            evalTransform = lambda x: x

        sbetas = []  # produce signed betas
        netweights = self.getNetworkWeights(onlyFF=True)
        for evc, nw in zip(evecs, netweights):
            nw = nw / torch.norm(nw, dim=1, keepdim=True)
            sbetas.append(nw.cpu() @ evc)

        ffLayers = [2, 3]
        shapedWeights = [[] for _ in range(len(ffLayers))]
        for idx in range(len(ffLayers)):
            assert np.all(evals[idx] >= 0), "Found negative eigenvalues..."
            # Compute fraction of variance explained by each eigenvector.
            cFractionVariance = evals[idx] / np.sum(evals[idx])

            # Make sure the datatype doesn't change to satisfy pytorch einsum.
            cKeepFraction = (evalTransform(cFractionVariance)
                             .astype(cFractionVariance.dtype))
            assert np.all(cKeepFraction >= 0), (
                   'Found negative transformed keep fractions. This means the '
                   'transform function has an improper form.')
            assert np.all(cKeepFraction <= 1), (
                   'Found keep fractions greater than 1. This is bad practice, '
                   'design the evalTransform function to have a domain and '
                   'range within [0,1]')

            # Measure norm of weights (this will be invariant to the change).
            weightNorms = torch.norm(netweights[idx], dim=1, keepdim=True)

            # Create tensor composed of each eigenvector scaled to its
            # contribution in each weight vector.
            evecComposition = torch.einsum('oi,xi->oxi', sbetas[idx],
                                           torch.tensor(evecs[idx]))

            # Scale eigenvectors based on their keep fraction (by default
            # scale them by their variance).
            newComposition = torch.einsum('oxi,i->ox', evecComposition,
                                          torch.tensor(cKeepFraction)).to(DEVICE)
            shapedWeights[idx] = (newComposition
                                  / torch.norm(newComposition, dim=1,
                                               keepdim=True)
                                  * weightNorms)

        # Assign new weights to network
        self.fc1.weight.data = shapedWeights[0]
        self.o.weight.data = shapedWeights[1]

    @staticmethod
    def targetedDropout(net, x, idx=None, layer=None, returnFull=False):
        assert layer >= 0 and layer <= 2, (
               'dropout only works on first three layers')
        c1 = net.convActivation(net.conv1(x))
        if layer == 0:
            fracDropout = len(idx) / c1.shape[1]
            c1[:, idx] = 0
            c1 = c1 * (1 - fracDropout)
        c2 = net.maxPool(net.convActivation(net.conv2(c1)))
        if layer == 1:
            fracDropout = len(idx) / c2.shape[1]
            c2[:, idx] = 0
            c2 = c2 * (1 - fracDropout)
        f1 = net.linearActivation(net.fc1(torch.flatten(c2, 1)))
        if layer == 2:
            fracDropout = len(idx) / f1.shape[1]
            f1[:, idx] = 0
            f1 = f1 * (1 - fracDropout)
        out = net.o(f1)
        if returnFull:
            return c1, c2, f1, out
        else:
            return out

    @staticmethod
    def mlTargetedDropout(net, x, idx, layer, returnFull=False):

        assert type(idx) is tuple and type(layer) is tuple, (
               'idx and layer need to be tuples')
        assert len(idx) == len(layer), (
               'idx and layer need to have the same length')
        npLayer = np.array(layer)
        assert len(npLayer) == len(np.unique(npLayer)), (
               'layer must not have any repeated elements')

        # Do forward pass with targeted dropout.
        c1 = net.convActivation(net.conv1(x))
        if np.any(npLayer == 0):
            cIndex = idx[npLayer == 0]
            fracDropout = len(cIndex) / c1.shape[1]
            c1[:, cIndex] = 0
            c1 = c1 * (1 - fracDropout)
        c2 = net.maxPool(net.convAcivation(net.conv2(c1)))
        if np.any(npLayer == 1):
            cIndex = idx[npLayer == 1]
            fracDropout = len(cIndex) / c2.shape[1]
            c2[:, cIndex] = 0
            c2 = c21 * (1 - fracDropout)
        f1 = net.linearActivation(net.fc1(torch.flatten(c2, 1)))
        if np.any(npLayer == 2):
            cIndex = idx[npLayer == 2]
            fracDropout = len(cIndex) / f1.shape[1]
            f1[:, cIndex] = 0
            f1 = f1 * (1 - fracDropout)
        out = net.o(f1)
        if returnFull:
            return c1, c2, f1, out
        else:
            return out

    @staticmethod
    def inputEigenfeatures(net, dataloader, onlyFF=True, DEVICE=None):
        # Handle DEVICE if not provided
        if DEVICE is None:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Measure Activations (without dropout) for all images
        storeDropout = net.getDropout()
        net.setDropout(0)  # no dropout for measuring eigenfeatures
        allimages = []
        activations = []
        for images, label in dataloader:
            allimages.append(images)
            images = images.to(DEVICE)
            label = label.to(DEVICE)
            activations.append(net.getActivations(images))
        net.setDropout(storeDropout)

        # Consolidate variable structure
        allinputs = []

        # Only add inputs to convolutional layers if onlyFF switch is off.
        if not onlyFF:

            # Inputs to first convolutional layer.
            allinputs.append(torch.flatten(torch.cat(allimages, dim=0)
                                                .detach()
                                                .cpu(), 1))

            # Inputs to second convolutional layer.
            allinputs.append(torch.flatten(torch.cat([cact[0] for
                                                      cact in activations], dim=0)
                                                .detach()
                                                .cpu(), 1))

        # Inputs to first feedforward layer.
        allinputs.append(torch.flatten(torch.cat([cact[1] for cact
                                                  in activations], dim=0)
                                            .detach()
                                            .cpu(), 1))

        # Inputs to last convolutional layer.
        allinputs.append(torch.cat([cact[2] for cact in activations], dim=0)
                              .detach()
                              .cpu())

        # Measure eigenfeatures for input to each feedforward layer.
        eigenvalues = []
        eigenvectors = []
        for ai in allinputs:
            # Covariance matrix is positive semidefinite, but numerical errors
            # can produce negative eigenvalues.
            ccov = torch.cov(ai.T)
            crank = torch.linalg.matrix_rank(ccov)
            w, v = sp.linalg.eigh(ccov)
            widx = np.argsort(w)[::-1]
            w = w[widx]
            v = v[:, widx]
            # Automatically set eigenvalues to 0 when they are numerical
            # errors!
            w[crank:] = 0
            eigenvalues.append(w)
            eigenvectors.append(v)

        return eigenvalues, eigenvectors

    @staticmethod
    def measureEigenFeatures(net, dataloader, onlyFF=True, DEVICE=None):
        eigenvalues, eigenvectors = CNN2P2.inputEigenfeatures(net,
                                                              dataloader,
                                                              onlyFF=onlyFF,
                                                              DEVICE=DEVICE)

        # Measure dot product of weights on eigenvectors for each layer.
        beta = []
        netweights = net.getNetworkWeights(onlyFF=onlyFF)
        for evc, nw in zip(eigenvectors, netweights):
            nw = nw / torch.norm(nw, dim=1, keepdim=True)
            beta.append(torch.abs(nw.cpu() @ evc))

        return beta, eigenvalues, eigenvectors

    @staticmethod
    def avg_within_layer(full):

        '''
        Average across weights in each layer for each training step.
        Args:
            full:
                List of length n_train_steps, containing nested lists with
                metric per layer per weight.
        Returns:
            avgFull:
                num_laybers x n_train_steps, where metric has been averaged
                across weights in the layer.

        '''
        n_steps = len(full)
        n_layers = len(full[0])
        avgFull = torch.zeros((n_layers, n_steps))
        for layer in range(n_layers):
            avgFull[layer, :] = torch.tensor([torch.mean(f[layer])
                                              for f in full])
        return avgFull.cpu()

    @staticmethod
    def layerFromFull(full, layer, dim=1):
        if dim == 1:
            return torch.cat([f[layer][:, None] for f in full], dim=dim).cpu()
        elif dim == 2:
            return torch.cat([f[layer][:, :, None] for f in full], dim=dim).cpu()
        else:
            raise ValueError("Haven't coded layerFromFull for dimensions other than 1 or 2!")


class MLP4(nn.Module):
    """
    MLP with 4 layers
    Activation function is Relu by default (but can be chosen with
    hiddenactivation).
    Output activation function is identity, because we're using
    CrossEntropyLoss.
    """
    def __init__(self, actFunc=F.relu, pDropout=0.5):
        super().__init__()
        self.numLayers = 4
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=pDropout)

    def forward(self, x):
        self.hidden1 = self.actFunc(self.fc1(x))
        self.hidden2 = self.actFunc(self.fc2(self.dropout(self.hidden1)))
        self.hidden3 = self.actFunc(self.fc3(self.dropout(self.hidden2)))
        self.output = self.fc4(self.dropout(self.hidden3))
        return self.output

    def setDropout(self, pDropout):
        self.dropout.p = pDropout

    def getDropout(self):
        return self.dropout.p

    def doOjaUpdate(self, x, alpha):
        # Rule: dW = alpha * (xy - wy**2)
        B = x.shape[0]
        activations = self.getActivations(x)
        # Layer 1:
        H, D = (activations[0].shape[1], x.shape[1])
        dfc1 = (alpha
                * (activations[0].T @ x
                   - torch.sum(self.fc1.weight.data.clone().detach().reshape(H, D, 1)
                               * (activations[0] * 2).T.reshape(H, B, 1).permute(0, 2, 1), dim=2)
                   )
                )
        self.fc1.weight.data = self.fc1.weight.data + dfc1
        self.fc1.weight.data = (self.fc1.weight.data
                                / torch.norm(self.fc1.weight.data, dim=1,
                                             keepdim=True))
        #print(f"fc1: Weight.shape:{self.fc1.weight.data.shape}, update.shape:{dfc1.shape}")

        # Layer 2:
        H, D = (activations[1].shape[1], activations[0].shape[1])
        dfc2 = alpha * (activations[1].T @ activations[0] - torch.sum(self.fc2.weight.data.clone().detach().reshape(H, D, 1) * (activations[1] * 2).T.reshape(H, B, 1).permute(0, 2, 1), dim=2))
        self.fc2.weight.data = self.fc2.weight.data + dfc2
        self.fc2.weight.data = self.fc2.weight.data / torch.norm(self.fc2.weight.data, dim=1, keepdim=True)
        #print(f"fc2: Weight.shape:{self.fc2.weight.data.shape}, update.shape:{dfc2.shape}")
        # Layer 3:
        H, D = (activations[2].shape[1], activations[1].shape[1])
        dfc3 = alpha * (activations[2].T @ activations[1] - torch.sum(self.fc3.weight.data.clone().detach().reshape(H, D, 1) * (activations[2] * 2).T.reshape(H, B, 1).permute(0, 2, 1), dim=2))
        self.fc3.weight.data = self.fc3.weight.data + dfc3
        self.fc3.weight.data = self.fc3.weight.data / torch.norm(self.fc3.weight.data, dim=1, keepdim=True)
        #print(f"fc3: Weight.shape:{self.fc3.weight.data.shape}, update.shape:{dfc3.shape}")
        # Layer 4:
        H, D = (activations[3].shape[1], activations[2].shape[1])
        dfc4 = alpha * (activations[3].T @ activations[2] - torch.sum(self.fc4.weight.data.clone().detach().reshape(H, D, 1) * (activations[3] * 2).T.reshape(H, B, 1).permute(0, 2, 1), dim=2))
        self.fc4.weight.data = self.fc4.weight.data + dfc4
        self.fc4.weight.data = self.fc4.weight.data / torch.norm(self.fc4.weight.data, dim=1, keepdim=True)
        #print(f"fc4: Weight.shape:{self.fc4.weight.data.shape}, update.shape:{dfc4.shape}")

    @torch.no_grad()
    def getActivations(self, x, usePrevious=False):
        if not usePrevious:
            _ = self.forward(x)
        activations = [
            self.hidden1,
            self.hidden2,
            self.hidden3,
            self.output
        ]
        return activations

    @torch.no_grad()
    def getNetworkWeights(self):
        # with torch.no_grad():
        netWeights = [param.clone() for name, param in self.named_parameters()
                      if 'weight' in name]
        # netWeights = []
        # netWeights.append(self.fc1.weight.data.clone().detach())
        # netWeights.append(self.fc2.weight.data.clone().detach())
        # netWeights.append(self.fc3.weight.data.clone().detach())
        # netWeights.append(self.fc4.weight.data.clone().detach())
        return netWeights

    @torch.no_grad()
    def compareNetworkWeights(self, initWeights):
        currWeights = self.getNetworkWeights()
        deltaWeights = []
        for iw, cw in zip(initWeights, currWeights):
            iw = torch.flatten(iw, 1)
            cw = torch.flatten(cw, 1)
            deltaWeights.append(torch.norm(cw - iw, dim=1))
        return deltaWeights

    @torch.no_grad()
    def measureAlignment(self, x, usePrevious=False, method='alignment'):
        if method == 'alignment':
            align_func = aat.alignmentLinearLayer
        elif method == 'similarity':
            align_func = aat.similarityLinearLayer
        activations = self.getActivations(x, usePrevious=usePrevious)
        alignment = [
            align_func(x, self.fc1),
            align_func(activations[0], self.fc2),
            align_func(activations[1], self.fc3),
            align_func(activations[2], self.fc4)
        ]
        return alignment

    def manualShape(self, evals, evecs, DEVICE, evalTransform=None):

        if evalTransform is None:
            evalTransform = lambda x: x

        sbetas = []  # produce signed betas
        netweights = self.getNetworkWeights()
        for evc, nw in zip(evecs, netweights):
            nw = nw / torch.norm(nw, dim=1, keepdim=True)
            sbetas.append(nw.cpu() @ evc)

        shapedWeights = [[] for _ in range(self.numLayers)]
        for layer in range(self.numLayers):
            assert np.all(evals[layer] >= 0), "Found negative eigenvalues..."
            # Compute fraction of variance explained by each eigenvector.
            cFractionVariance = evals[layer] / np.sum(evals[layer])

            # Make sure the datatype doesn't change to satisfy pytorch einsum.
            cKeepFraction = (evalTransform(cFractionVariance)
                             .astype(cFractionVariance.dtype))
            assert np.all(cKeepFraction >= 0), (
                   'Found negative transformed keep fractions. This means the '
                   'transform function has an improper form.')
            assert np.all(cKeepFraction <= 1), (
                   'Found keep fractions greater than 1. This is bad '
                   'practice, design the evalTransform function to have a '
                   'domain and range within [0,1]')

            # Measure norm of weights (this will be invariant to the change).
            weightNorms = torch.norm(netweights[layer], dim=1, keepdim=True)

            # Create tensor composed of each eigenvector scaled to its
            # contribution in each weight vector.
            evecComposition = torch.einsum('oi,xi->oxi', sbetas[layer],
                                           torch.tensor(evecs[layer]))

            # Scale eigenvectors based on their keep fraction (by default
            # scale them by their variance).
            newComposition = torch.einsum('oxi,i->ox',
                                          evecComposition,
                                          torch.tensor(cKeepFraction)).to(DEVICE)
            shapedWeights[layer] = (newComposition
                                    / torch.norm(newComposition, dim=1,
                                                 keepdim=True)
                                    * weightNorms)

        # Assign new weights to network
        self.fc1.weight.data = shapedWeights[0]
        self.fc2.weight.data = shapedWeights[1]
        self.fc3.weight.data = shapedWeights[2]
        self.fc4.weight.data = shapedWeights[3]

    @staticmethod
    def targetedDropout(net, x, idx=None, layer=None, returnFull=False):
        assert layer >= 0 and layer <= 2, (
               'dropout only works on first three layers')
        h1 = net.actFunc(net.fc1(x))
        if layer == 0:
            fracDropout = len(idx) / h1.shape[1]
            h1[:, idx] = 0
            h1 = h1 * (1 - fracDropout)
        h2 = net.actFunc(net.fc2(h1))
        if layer == 1:
            fracDropout = len(idx) / h2.shape[1]
            h2[:, idx] = 0
            h2 = h2 * (1 - fracDropout)
        h3 = net.actFunc(net.fc3(h2))
        if layer == 2:
            fracDropout = len(idx) / h3.shape[1]
            h3[:, idx] = 0
            h3 = h3 * (1 - fracDropout)
        out = net.fc4(h3)
        if returnFull:
            return h1, h2, h3, out
        else:
            return out

    @staticmethod
    def mlTargetedDropout(net, x, idx, layer, returnFull=False):
        assert type(idx) is tuple and type(layer) is tuple, (
               'idx and layer need to be tuples')
        assert len(idx) == len(layer), (
               'idx and layer need to have the same length')
        npLayer = np.array(layer)
        assert len(npLayer) == len(np.unique(npLayer)), (
               'layer must not have any repeated elements')

        # Do forward pass with targeted dropout.
        h1 = net.actFunc(net.fc1(x))
        if np.any(npLayer == 0):
            cIndex = idx[npLayer == 0]
            fracDropout = len(cIndex) / h1.shape[1]
            h1[:, cIndex] = 0
            h1 = h1 * (1 - fracDropout)
        h2 = net.actFunc(net.fc2(h1))

        if np.any(npLayer == 1):
            cIndex = idx[npLayer == 1]
            fracDropout = len(cIndex) / h2.shape[1]
            h2[:, cIndex] = 0
            h2 = h2 * (1 - fracDropout)
        h3 = net.actFunc(net.fc3(h2))

        if np.any(npLayer == 2):
            cIndex = idx[npLayer == 2]
            fracDropout = len(cIndex) / h3.shape[1]
            h3[:, cIndex] = 0
            h3 = h3 * (1 - fracDropout)
        out = net.fc4(h3)

        if returnFull:
            return h1, h2, h3, out
        else:
            return out

    @staticmethod
    @torch.no_grad()
    def measureEigenFeatures(net, dataloader, DEVICE=None):
        # Handle DEVICE if not provided
        if DEVICE is None:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Measure Activations (without dropout) for all images.
        storeDropout = net.getDropout()
        net.setDropout(0)  # no dropout for measuring eigenfeatures
        allimages = []
        activations = []
        for images, labels in dataloader:
            allimages.append(images)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            activations.append(net.getActivations(images))
        net.setDropout(storeDropout)

        # Consolidate variable structure.
        n_layers = net.numLayers
        allinputs = []
        allinputs.append(torch.cat(allimages, dim=0).detach().cpu())
        for layer in range(n_layers - 1):
            allinputs.append(torch
                             .cat([cact[layer] for cact in activations],
                                       dim=0)
                             .detach()
                             .cpu())

        # Measure eigenfeatures for each layer.
        eigenvalues = []
        eigenvectors = []
        for ai in allinputs:
            # Covariance matrix is positive semidefinite, but numerical errors
            # can produce negative eigenvalues.
            ccov = torch.cov(ai.T)
            crank = torch.linalg.matrix_rank(ccov)
            w, v = sp.linalg.eigh(ccov)
            widx = np.argsort(w)[::-1]
            w = w[widx]
            v = v[:, widx]
            # Automatically set eigenvalues to 0 when they are numerical
            # errors!
            w[crank:] = 0
            eigenvalues.append(w)
            eigenvectors.append(v)

        # Measure dot product of weights on eigenvectors for each layer.
        beta = []
        netweights = net.getNetworkWeights()
        for evc, nw in zip(eigenvectors, netweights):
            nw = nw / torch.norm(nw, dim=1, keepdim=True)
            beta.append(torch.abs(nw.cpu() @ evc))

        return beta, eigenvalues, eigenvectors

    @staticmethod
    @torch.no_grad()
    def avg_within_layer(full):

        '''
        Average across weights in each layer for each training step.
        Args:
            full:
                List of length n_train_steps, containing nested lists with
                metric per layer per weight.
        Returns:
            avgFull:
                num_laybers x n_train_steps, where metric has been averaged
                across weights in the layer.

        '''
        n_steps = len(full)
        n_layers = len(full[0])
        avgFull = torch.zeros((n_layers, n_steps))
        for layer in range(n_layers):
            avgFull[layer, :] = torch.tensor([torch.mean(f[layer])
                                              for f in full])
        return avgFull.cpu()

    @staticmethod
    @torch.no_grad()
    def layerFromFull(full, layer, dim=1):
        if dim == 1:
            return torch.cat([f[layer][:, None] for f in full], dim=dim).cpu()
        elif dim == 2:
            return torch.cat([f[layer][:, :, None] for f in full], dim=dim).cpu()
        else:
            raise ValueError("Haven't coded layerFromFull for dimensions other than 1 or 2!")


class AlexNet(nn.Module):
    """
    Local implementation of AlexNet to measure internal features during
    training without hooks.
    https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
    """
    def __init__(self, weights='DEFAULT', pDropout=0):
        super().__init__()
        alexnet = models.alexnet(weights=weights, progress=True)
        # Maybe better to write this as composed layers, with ReLU in place
        # for each appropriate layer?
        # Inputs are 224+ x 224+
        # Layer 0
        self.conv0 = alexnet.features[0]  # 11x11, stride=4, nkernels=64
        # --> output dim: (227-11)/4 +1 ~ 55 (55x55x64)
        # Layer 1
        self.maxPool1 = alexnet.features[2]  # 3x3, stride=2
        # --> output dim: (55-3)/2 +1 = 27 (27x27x64)
        self.conv1 = alexnet.features[3]  # 5x5, pad=2, nkernels=192
        # --> output dim : (27+2*2-5)/1 +1 = 27 (27x27x192)
        # Layer 2
        self.maxPool2 = alexnet.features[5]  # 3x3, stride=2
        # --> output dim: (27-3)/2 +1 = 13 (13x13x192)
        self.conv2 = alexnet.features[6]  # 3x3, pad=1, nkernels=384
        # --> output dim: (13+2*1-3)/2 +1 = 13 (13x13x384)
        # Layer 3
        self.conv3 = alexnet.features[8]  # 3x3, pad=1, nkernels=256
        # --> output dim: (13+2*1-3)/1 +1 = 13 (13x13x256)
        # Layer 4
        self.conv4 = alexnet.features[10]  # 3x3, pad=1, nkernels=256
        # --> output dim: (13+2*1-3)/1 +1 = 13 (13x13x256)
        # Layer 5
        self.maxPool5 = alexnet.features[12]  # 3x3, stride=2
        # --> ouptut dim: (13-3)/2 +1 = 6 (6x6x256 = 9216 elements)
        self.avgpool5 = alexnet.avgpool  # output dim: 6x6x256
        self.do5 = alexnet.classifier[0]  # default p=0.5
        self.fc5 = alexnet.classifier[1]  # (6x6x256, 4096)
        # Layer 6
        self.do6 = alexnet.classifier[3]  # default p=0.5
        self.fc6 = alexnet.classifier[4]  # (4096, 4096)
        # Layer 7
        self.fc7 = alexnet.classifier[6]  # (4096, num_classes=1000)

        self.setDropout(pDropout)  # modify from defaults in pytorch model

    def forward(self, x):
        self.h0 = F.relu(self.conv0(x))
        self.h1 = F.relu(self.conv1(self.maxPool1(self.h0)))
        self.h2 = F.relu(self.conv2(self.maxPool2(self.h1)))
        self.h3 = F.relu(self.conv3(self.h2))
        self.h4 = F.relu(self.conv4(self.h3))
        self.h5 = F.relu(self.fc5(torch.flatten(self.do5(self.avgpool5(self.maxPool5(self.h4))), 1)))
        self.h6 = F.relu(self.fc6(self.do6(self.h5)))
        self.out = self.fc7(self.h6)
        return self.out

    def getDropout(self):
        return (self.do5.p, self.do6.p)

    def setDropout(self, pDropout):
        if isinstance(pDropout, (float, int)):
            pDropout = (pDropout, pDropout)
        self.do5.p = pDropout[0]
        self.do6.p = pDropout[1]

    @torch.no_grad()
    def getActivations(self, x, ffOnly=False, usePrevious=False):
        if not usePrevious:
            _ = self.forward(x)
        activations = [
            self.h0,
            self.h1,
            self.h2,
            self.h3,
            self.h4,
            self.h5,
            self.h6,
            self.out
        ]
        return activations

    @torch.no_grad()
    def getNetworkWeights(self, ffOnly=False):
        netWeights = []
        if not ffOnly:
            netWeights.append(self.conv0.weight.data.clone().detach())
            netWeights.append(self.conv1.weight.data.clone().detach())
            netWeights.append(self.conv2.weight.data.clone().detach())
            netWeights.append(self.conv3.weight.data.clone().detach())
            netWeights.append(self.conv4.weight.data.clone().detach())
        netWeights.append(self.fc5.weight.data.clone().detach())
        netWeights.append(self.fc6.weight.data.clone().detach())
        netWeights.append(self.fc7.weight.data.clone().detach())
        return netWeights

    @torch.no_grad()
    def compareNetworkWeights(self, initWeights, ffOnly=False):
        currWeights = self.getNetworkWeights(ffOnly=ffOnly)
        deltaWeights = []
        for iw, cw in zip(initWeights, currWeights):
            iw = torch.flatten(iw, 1)
            cw = torch.flatten(cw, 1)
            deltaWeights.append(torch.norm(iw - cw, dim=1))
        return deltaWeights

    @torch.no_grad()
    def measureSimilarity(self, x):
        activations = self.getActivations(x)
        similarity = [torch.mean(aat.similarityConvLayer(x, self.conv0), axis=1),
                      torch.mean(aat.similarityConvLayer(self.maxPool1(activations[0]), self.conv1), axis=1),
                      torch.mean(aat.similarityConvLayer(self.maxPool2(activations[1]), self.conv2), axis=1),
                      torch.mean(aat.similarityConvLayer(activations[2], self.conv3), axis=1),
                      torch.mean(aat.similarityConvLayer(activations[3], self.conv4), axis=1),
                      aat.similarityLinearLayer(torch.flatten(self.do5(self.avgpool5(self.maxPool5(activations[4]))), 1), self.fc5),
                      aat.similarityLinearLayer(activations[5], self.fc6),
                      aat.similarityLinearLayer(activations[6], self.fc7)
                      ]
        return similarity

    @torch.no_grad()
    def measureAlignment(self, x, eachLook=True, usePrevious=False):
        activations = self.getActivations(x, usePrevious=usePrevious)
        if eachLook:
            alignment = [torch.mean(aat.alignmentConvLayer(x, self.conv0), axis=1),
                         torch.mean(aat.alignmentConvLayer(self.maxPool1(activations[0]), self.conv1), axis=1),
                         torch.mean(aat.alignmentConvLayer(self.maxPool2(activations[1]), self.conv2), axis=1),
                         torch.mean(aat.alignmentConvLayer(activations[2], self.conv3), axis=1),
                         torch.mean(aat.alignmentConvLayer(activations[3], self.conv4), axis=1)
                         ]
        else:
            alignment = [aat.alignmentConvLayer(x, self.conv0, eachLook=False),
                         aat.alignmentConvLayer(self.maxPool1(activations[0]), self.conv1, eachLook=False),
                         aat.alignmentConvLayer(self.maxPool2(activations[1]),
                                                    self.conv2,
                                                    eachLook=False),
                         aat.alignmentConvLayer(activations[2], self.conv3, eachLook=False),
                         aat.alignmentConvLayer(activations[3], self.conv4, eachLook=False)
                         ]
        alignment.append(aat.alignmentLinearLayer(torch.flatten(self.do5(self.avgpool5(self.maxPool5(activations[4]))), 1), self.fc5))
        alignment.append(aat.alignmentLinearLayer(activations[5], self.fc6))
        alignment.append(aat.alignmentLinearLayer(activations[6], self.fc7))
        return alignment

    @staticmethod
    def targetedDropout(net, x, idx=None, layer=None, returnFull=False):
        assert layer >= 0 and layer <= 6, (
               'dropout only works on first 7 layers')

        # Layer 0
        h0 = F.relu(net.conv0(x))
        if layer == 0:
            h0[:, idx] = 0

        # Layer 1
        h1 = F.relu(net.conv1(net.maxPool1(h0)))
        if layer == 1:
            h1[:, idx] = 0

        # Layer 2
        h2 = F.relu(net.conv2(net.maxPool2(h1)))
        if layer == 2:
            h2[:, idx] = 0

        # Layer 3
        h3 = F.relu(net.conv3(h2))
        if layer == 3:
            h3[:, idx] = 0

        # Layer 4
        h4 = F.relu(net.conv4(h3))
        if layer == 4:
            h4[:, idx] = 0

        # Layer 5
        h5 = F.relu(net.fc5(torch.flatten(net.do5(net.avgpool5(net.maxPool5(h4))), 1)))
        if layer == 5:
            h5[:, idx] = 0

        # Layer 6
        h6 = F.relu(net.fc6(net.do6(h5)))
        if layer == 6:
            h6[:, idx] = 0

        # Output
        out = net.fc7(h6)

        if returnFull:
            return h0, h1, h2, h3, h4, h5, h6, out
        else:
            return out

    @staticmethod
    def avg_within_layer(full):

        '''
        Average across weights in each layer for each training step.
        Args:
            full:
                List of length n_train_steps, containing nested lists with
                metric per layer per weight.
        Returns:
            avgFull:
                num_laybers x n_train_steps, where metric has been averaged
                across weights in the layer.

        '''
        n_steps = len(full)
        n_layers = len(full[0])
        avgFull = torch.zeros((n_layers, n_steps))
        for layer in range(n_layers):
            avgFull[layer, :] = torch.tensor([torch.mean(f[layer])
                                              for f in full])
        return avgFull.cpu()

    @staticmethod
    def layerFromFull(full, layer):
        return torch.cat([f[layer].reshape(-1, 1) for f in full], dim=1).cpu()


def initWeights(imEvec, imEval, numNeurons, scaleFunction=None):
    numEval = len(imEval)
    if scaleFunction is not None:
        imEval = scaleFunction(imEval)
    initBeta = np.random.exponential(scale=imEval, size=(numNeurons, numEval)).astype('float32')
    signBeta = (np.random.randint(0, 2, size=initBeta.shape).astype('float32')
                * 2 - 1)
    beta = torch.tensor(initBeta * signBeta)
    weights = imEvec @ beta.T
    return weights
