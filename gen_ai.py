import time
import os

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as TT
from torchsummary import summary

import numpy as np 
import matplotlib.pyplot as plt

from vll.train.VAE_utils import train, validate, plot_training
from typing import List, Optional


# ========================= PARAMETERS ==============================
# The model to be used.  Possible values: 'AE' , 'VAE'
model_type = 'VAE'
layer_dimensions = [32, 64, 128, 256]
latent_dimension = 2

# If the model should only run the health checks and not train
check_health = False
verbose = True
# ================= SAVE TO DISK ================

write_to_disk = True

# ================= SAVE TO DISK ================
plot_progress = False
plot_interval = 15

epoches = 100
learning_rate = 0.2

#log_interval * batchsize = how often the intermediate result is displayed
batch_size = 32
log_interval = 100

test_batch_size = 1000

# Activation Function === possible values : 'leakyrelu', 'sigmoid', 'tanh', 'silu'
activation_func = 'leakyrelu'
# LeakyReLU leak value
leak = 0

# Loss Function
kl_loss_weight = 0.5
l1_loss_weight = 0.0
mse_loss_weight = 0.5

if write_to_disk == True:
    root_directory = '/home/student/Documents/IWR-CVL/3dcv-students/gen_ai-Models/Version-%d/'
else:
    root_directory = '/tmp/IWR-CVL-Trash-%d/'

i=0
while(1):
    try:
        os.mkdir(root_directory % i)
        root_directory = str(root_directory % i)
        print("Directory", root_directory, ' has been created.')
        break
    except FileExistsError:
        print("WARNING: Directory", root_directory % i, ' already exists.')
        i = i + 1



# ========================= PARAMETERS END ==============================

# use GPU if available
use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# initialize data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, 
    transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

print('RUNNING CUDA' if use_cuda else 'RUNNING CPU')
print('=== PARAMETERS ===')
print('Model: ', model_type)
print('Layer Dimensions:', layer_dimensions)
print('Latent Dimensions:', latent_dimension)
print('Epoches: ', epoches)
print('Batch Size: ', batch_size)
print('Learning Rate: ', learning_rate)
print('Activation Function: ', activation_func)
if activation_func == 'leakyrelu': print('ReLU Leak: ', leak)
print('Verbose: ', verbose)
print('Plot Progress: ', plot_progress)
print('Plot Interval: ', plot_interval)
print('Only Check Health: ', check_health)
print('Write to disk: ', write_to_disk)

# ================== HELPER FUNCTIONS ==========================================
#takes an array as the input and saves it at the specified path as a cvs file
def arr2cvs_file(filename: str, array, width: int, *label):  
    '''
    Takes an 2D array as an input and saves it to a cvs file.
    The shape must be consistent. And the width specified.
    '''
    try:
        with open(root_directory + filename, 'w') as file:
            heading = str(label[0])
            for i in range(width-1):
                heading = heading + ',' + str(label[i+1])

            file.write(heading + '\n')
            for i, item in enumerate(array):

                entry = str(item[0])

                for y in range(width-1):
                    entry = entry + ',' + str(item[y+1])

                file.write( entry + '\n' )

    except:
        print("Error: Something went wrong while saving to cvs")
    pass

# ========================= INITIALIZATION ================================

start_time = time.time()

parameters = []
parameters.append(['Model', model_type])
list_layer_dimensions = str(layer_dimensions[0])
for i in range(len(layer_dimensions) - 1):
    list_layer_dimensions = list_layer_dimensions + ':' + str(layer_dimensions[i + 1])
parameters.append(['Layers, En/Decoder mirrored', list_layer_dimensions])
parameters.append(['Latent Layer Dimension', latent_dimension])
parameters.append(['Epoches', epoches])
parameters.append(['Batch Size', batch_size])
parameters.append(['Test Batch Size', test_batch_size])
parameters.append(['Learning Rate', learning_rate])
parameters.append(['Activation Function', activation_func])
if activation_func == 'leakyrelu': parameters.append(['ReLU Leak', leak])
parameters.append(['KL Loss Weight', kl_loss_weight])
parameters.append(['L1 Loss Weight', l1_loss_weight])
parameters.append(['MSE Loss Weight', mse_loss_weight])

arr2cvs_file('parameters.cvs', parameters, 2, 'Parameter', 'Value')

# ========================= INITIALIZATION ================================


# ================== Vanilla Encoder Class =====================================

class Vanilla_AE(nn.Module):
    '''
    Implementation of a Simple AE
    -----------------------------
    in_channels: Number of input Channels (in our case number of image Channels)
    latent_dim: determines the size of the latent dimension
    input_size: np.array of H x W of input images 
    hidden_dim: List of channel sizes of the Encoder Decoder layers
    

    -----------------------------
    Will setup a simple AE with a encode, decode, forward, loss_function, sample and generate function
    '''

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: np.ndarray,
                 hidden_dims: List = None,
                 activation_function = None,
                 **kwargs) -> None:
    
        super(Vanilla_AE, self).__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.leak = leak
        self.activ_func = activation_function
    
        modules = []
        if hidden_dims is None:
            return 0
        else: 
            self.hidden_dims = hidden_dims

        # TODO: Build Encoder
        # Tipp: You can use a loop over the hidden dims in combination with a nn.Sequential


        # TODO: Save complete Encoder Sequential as self.encoder
        #layers = self.hidden_dims


        #self.encoder = nn.Sequential(
        #    nn.Conv2d(1, layers[0], 3, 2, 1),
        #    nn.BatchNorm2d(layers[0]),
        #    nn.LeakyReLU(self.leak),

        #    nn.Conv2d(layers[0], layers[1], 3, 2, 1),
        #    nn.BatchNorm2d(layers[1]),
        #    nn.LeakyReLU(self.leak),

        #    nn.Conv2d(layers[1], layers[2], 3, 2, 1),
        #    nn.BatchNorm2d(layers[2]),
        #    nn.LeakyReLU(self.leak),

        #    nn.Conv2d(layers[2], layers[3], 3, 2, 1),
        #    nn.BatchNorm2d(layers[3]),
        #    nn.LeakyReLU(self.leak)
        #)
        # Create List for the parameters
        self.structure = []

        #modify the hidden_dims to contain all needed layers including the input/output Layer
        self.layer_struct = self.hidden_dims.copy()
        self.layer_struct.insert(0, in_channels) # input_channels
        #self.layer_struct.append(latent_dim) # Latent dimension

        #create a loop that loops over all the layer info and appends it to the list
        for i in range(len(self.layer_struct) - 1):
            self.structure.append(nn.Conv2d(self.layer_struct[i], self.layer_struct[i+1], 3, 2, 1))
            self.structure.append(nn.BatchNorm2d(self.layer_struct[i+1]))
            # If statement to choose which activation function to use. if Input is invalid ReLU is used
            if self.activ_func == 'leakyrelu':
                self.structure.append(nn.LeakyReLU(self.leak))
            elif self.activ_func == 'sigmoid':
                self.structure.append(nn.Sigmoid())
            elif self.activ_func == 'tanh':
                self.structure.append(nn.Tanh())
            elif self.activ_func == 'silu':
                self.structure.append(nn.SiLU())
            else:
                print("Encoder -> Activationfunction -> Using default ReLU <- Invalid Activation function")
                self.structure.append(nn.ReLU())

        #encode it all inside the nn.Sequential()
        #print(self.structure)
        self.encoder = nn.Sequential(*self.structure)

        # We basically calculate the image resolution here 
        # Since we use a fixed latent dim we need to calculate what the latent dim will be with our 
        # fixed image size. Since we always use stride=2, with each block we reduce the image resolution
        # by 2. 
        # Because // always rounds down, while nn.Conv2D will round up we add 2 - 1 to get the rounded up 
        # result.
        self.block_dimension = self.input_size 
        #print(self.block_dimension)
        for i in range(len(self.hidden_dims)): 
            self.block_dimension = (self.block_dimension + 2 - 1 ) // 2 

        # This is used for the VAE later on 
        # For the AE we just use the fully_connecte mu predictor (fc_mu)
        #print(type(self.block_dimension))
        #print(self.block_dimension)
        #print(self.block_dimension)
        #print(self.hidden_dims[-1] * self.block_dimension[0] * self.block_dimension[1])
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * self.block_dimension[0] * self.block_dimension[1], latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * self.block_dimension[0] * self.block_dimension[1], latent_dim)


        # Build Decoder
        # Since the upsample results in images of 32 by 32 we need a resize operation in the final_layer
        # Remember to do this
        # TODO: Implement your decoder
        # You can do this similar to the encoder


        # TODO: Save complete Decoder Sequential as self.Decoder
        #layers = self.hidden_dims
        self.decoder_ff = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dims[-1] * self.block_dimension[0] * self.block_dimension[1]),
            nn.ReLU()
        )

        #self.decoder = nn.Sequential(
            
        #    nn.ConvTranspose2d(layers[3], layers[2], 3, 2, 1, 1),
        #    nn.BatchNorm2d(layers[2]),
        #    nn.LeakyReLU(self.leak),

        #    nn.ConvTranspose2d(layers[2], layers[1], 3, 2, 1, 1),
        #    nn.BatchNorm2d(layers[1]),
        #    nn.LeakyReLU(self.leak),

        #    nn.ConvTranspose2d(layers[1], layers[0], 3, 2, 1, 1),
        #    nn.BatchNorm2d(layers[0]),
        #    nn.LeakyReLU(self.leak),

        #    nn.ConvTranspose2d(layers[0], layers[0], 3, 2, 1, 1),
        #    nn.BatchNorm2d(layers[0]),
        #    nn.LeakyReLU(self.leak),
        #)

        self.structure = []
        #print(hidden_dims)

        #modify the hidden_dims to contain all needed layers including the input/output Layer
        self.layer_struct = self.hidden_dims.copy()
        #self.layer_struct.insert(0, in_channels) # input_channels
        #self.layer_struct.append(latent_dim*latent_dim) # Latent dimension

        self.layer_struct.reverse()
        self.layer_struct.append(self.layer_struct[-1])
        #print(self.layer_struct)

        #create a loop that loops over all the layer info and appends it to the list
        for i in range(len(self.layer_struct) - 1):
            self.structure.append(nn.ConvTranspose2d(self.layer_struct[i], self.layer_struct[i+1], 3, 2, 1, 1))
            self.structure.append(nn.BatchNorm2d(self.layer_struct[i+1]))
            # if statement to determine activation function
            if self.activ_func == 'leakyrelu':
                self.structure.append(nn.LeakyReLU(self.leak))
            elif self.activ_func == 'sigmoid':
                self.structure.append(nn.Sigmoid())
            elif self.activ_func == 'tanh':
                self.structure.append(nn.Tanh())
            elif self.activ_func == 'silu':
                self.structure.append(nn.SiLU())
            else:
                print("Decoder -> Activationfunction -> Using default ReLU <- Invalid Activation function")
                self.structure.append(nn.ReLU())

        #encode it all inside the nn.Sequential()
        #print(self.structure)
        self.decoder = nn.Sequential(*self.structure)


        # TODO: Set up Final Layer of Transposed Convolution and Resizing
        self.final_layer = nn.Sequential(
            TT.Resize(28),
            nn.Conv2d(self.layer_struct[-1], 1, 1, 1, 0),
            nn.Tanh()
        )


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # TODO: Implement the encoder forward
        # Use the self.fc_mu as well to predict the latent directly
        #print(input.shape)
        encoding = self.encoder(input)
        #print(encoding.shape)
        encoding = encoding.flatten(1)
        #print(encoding.shape)
        latent_code = self.fc_mu(encoding)

        return latent_code

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D] (D is the latent Dimension)
        :return: (Tensor) [B x C x H x W]
        """
        # TODO: Implement the decoder forward
        
        #print(z)
        z = self.decoder_ff(z)

        z = torch.reshape(z, (-1, self.hidden_dims[-1], 2, 2))
        decoding = self.decoder(z)
        reconstruction = self.final_layer(decoding)

        return reconstruction

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        ''' 
        Implemets the complete forward pass. 
        '''
        # TODO: Implemnt the combination of Encoder and Decoder
        # The outputs have to be in a list in the exact order given below
        z = self.encode(input)
        reconstruction = self.decode(z)


        return [reconstruction, input, z]

    def loss_function(self,
                      prediction: List[Tensor], # This is what you return in forward !
                      target: Tensor, 
                      **kwargs) -> dict:
        """
        Computes the AE loss function.
        AE: Just simple MSE_loss
        -----------------------------
        prediction: output of forward call 
        target: target image
        param kwargs: optional kwargs
        ------------------------------

        return: Loss
        """
        # TODO: Implement the AE loss function
        mse_loss = mse_loss_weight * nn.functional.mse_loss(prediction[0], target)
        l1_loss = l1_loss_weight * nn.functional.l1_loss(prediction[0], target)
        return mse_loss + l1_loss


# ================== VARIATIONAL Encoder Class VAE =====================================

class VAE(Vanilla_AE):
    '''
    Implementation of a Simple VAE
    -----------------------------
    in_channels: Number of input Channels (in our case number of image Channels)
    latent_dim: determines the size of the latent dimension
    input_size: np.array of H x W of input images 
    hidden_dim: List of channel sizes of the Encoder Decoder layers
    
    '''

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: np.ndarray,
                 hidden_dims: List = None,
                 activation_function = None,
                 **kwargs) -> None:
        super().__init__(in_channels, latent_dim, input_size, hidden_dims, activation_function)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # TODO: Implement the forward of the encode
        # This should return the predicted mu and log_var (use self.fc_mu and self.fc_var for this)
        # in the exact order given below

        encoding = self.encoder(input)
        encoding = encoding.flatten(1)
        mu = self.fc_mu(encoding)
        log_var = self.fc_var(encoding)

        return  [mu, log_var]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # TODO: Implement forward with reparametrization in it
        # Return in exact order given
        # ==================================================PROBLEM, 
        # how do I sample the point for the decoder and what is the reparametrization all about?
        latent_info = self.encode(input)
        mu, log_var = latent_info
        z = self.reparameterize(mu, log_var)

        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      prediction: List[Tensor],
                      target: Tensor,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        prediction: List with [recons, input, mu, log_var]
        target: Tensor,
        
        REMEMBER: You predict the log(sigma) not sigma
        """
        # TODO: Implement Loss as discribed above
        log_sigma = prediction[3]
        sigma = torch.exp(prediction[3])
        mu = prediction[2]
        num_batches = len(prediction[1])

        input = prediction[1].view(-1, 28*28)
        reconstruction = prediction[0].view(-1, 28*28)

        kl_loss = kl_loss_weight * torch.sum( mu * mu + sigma - log_sigma - 1) / num_batches 
        mse_loss = mse_loss_weight * torch.sum( torch.pow(input - reconstruction, 2), 1)
        l1_loss = l1_loss_weight * torch.sum( torch.abs(input - reconstruction), 1)

        loss = kl_loss + mse_loss.mean() + l1_loss.mean()

        return loss

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input latent z, returns the reconstructed image
        :param z: (Tensor) [B x latent_dim]
        :return: (Tensor) [B x C x H x W]
        """
        # Function used to generate samples out of a given latent

        return self.decode(x)[0]


# ===================================== VANILLA AUTO ENCODER Health Check ====================================================
# Test your implementation
# You should see the original MNIST images and 2 noisy images below (since the AE is not trained yet)
if model_type == 'AE':
    AE_model = Vanilla_AE(in_channels=1, latent_dim=latent_dimension, input_size=np.array([28, 28]), hidden_dims=layer_dimensions, activation_function=activation_func)

if model_type == 'AE' and check_health == True:
    summary(AE_model, input_size=(1, 28, 28), device='cpu')


    data_instance = iter(train_loader)
    images, labels = next(data_instance)


    predicted_img, input, z = AE_model(images)

    fig, axes = plt.subplots(2, 3, figsize=(10, 3))
    for i in range(3):
        image = images[i].numpy().squeeze()  # Bild in numpy-Array umwandeln und Kanal dimension entfernen
        axes[0][i].imshow(image, cmap='gray')
        axes[0][i].set_title(f'Label: {labels[i].item()}')
        axes[0][i].axis('off')
        # Same for prediction
        pred_img = predicted_img[i].detach().numpy().squeeze()
        axes[1][i].imshow(pred_img, cmap='gray')
        axes[1][i].set_title(f'Label: {labels[i].item()}')
        axes[1][i].axis('off')
    plt.show()


# ===================================== VARIATIONAL AUTO ENCODER Health Check ====================================================
# Test your implementation
# You should see the original MNIST images and 2 noisy images below (since the AE is not trained yet)
if model_type == 'VAE':
    VAE_model = VAE(in_channels=1, latent_dim=latent_dimension, input_size=np.array([28, 28]), hidden_dims=layer_dimensions, activation_function=activation_func)

if model_type == 'VAE' and check_health == True:
    summary(VAE_model, input_size=(1, 28, 28), device='cpu')


    data_instance = iter(train_loader)
    images, labels = next(data_instance)

    predicted_img, input, mu, log_var = VAE_model(images)

    fig, axes = plt.subplots(2, 3, figsize=(10, 3))
    for i in range(3):
        image = images[i].numpy().squeeze()  # Bild in numpy-Array umwandeln und Kanal dimension entfernen
        axes[0][i].imshow(image, cmap='gray')
        axes[0][i].set_title(f'Label: {labels[i].item()}')
        axes[0][i].axis('off')
        # Same for prediction
        pred_img = predicted_img[i].detach().numpy().squeeze()
        axes[1][i].imshow(pred_img, cmap='gray')
        axes[1][i].set_title(f'Label: {labels[i].item()}')
        axes[1][i].axis('off')
    plt.show()


# ========================== VANILLA Training ================================================================
if model_type == 'AE' and check_health == False:

    epochs = epoches
    lr = learning_rate
    log_interval = log_interval

    optimizer = optim.Adadelta(AE_model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


    tr_loss = []
    tr_loss_step = []
    test_loss = []
    if use_cuda: 
        AE_model = AE_model.cuda()
    for epoch in range(1, epochs + 1):
        # Validate 
        if epoch % plot_interval == 0:
            validate(model=AE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=plot_progress, verbose=verbose)
        else:
            validate(model=AE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=False, verbose=verbose)
        # train one epoch
        train_loss_tmp = []
        train(model=AE_model, use_cuda=use_cuda, train_loader=train_loader, optimizer=optimizer,
            epoch=epoch, log_interval=log_interval, tr_loss=train_loss_tmp, verbose=verbose)
        
        print('EPOCHE: ', epoch, '=== Elapsed time: ', time.time() - start_time, 'seconds')

        for i in range(len(train_loss_tmp)):
            tr_loss_step.append(train_loss_tmp[i])
        tr_loss.append(np.sum(train_loss_tmp).item() / len(train_loader.dataset))
        if verbose == True:
            print(f'Train Epoch: {epoch} \t Average: {tr_loss[-1]:.8f}')

        scheduler.step(test_loss[-1])

    training_loss = []
    if len(tr_loss) == len(test_loss):
        for i, train_loss in enumerate(tr_loss):
            training_loss.append([train_loss, test_loss[i]])

    arr2cvs_file('loss.cvs', training_loss, 2, 'Training Loss', 'Test Loss')

    plot_training(tr_loss_step, tr_loss, test_loss, epochs, train_loader, batch_size)
    validate(model=AE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=plot_progress, verbose=verbose)
    if write_to_disk == True:
        torch.save(AE_model.state_dict(), root_directory+'pytorch_model_weights.pt')


# ========================== VARIATIONAL Training ================================================================
if model_type == 'VAE' and check_health == False:
    epochs = epoches
    lr = learning_rate
    log_interval = log_interval

    optimizer = optim.Adadelta(VAE_model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


    tr_loss = []
    tr_loss_step = []
    test_loss = []
    if use_cuda: 
        VAE_model = VAE_model.cuda()
    for epoch in range(1, epochs + 1):
        # Validate 
        if epoch % plot_interval == 0:
            validate(model=VAE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=plot_progress, verbose=verbose)
        else:
            validate(model=VAE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=False, verbose=verbose)
        # train one epoch
        train_loss_tmp = []
        train(model=VAE_model, use_cuda=use_cuda, train_loader=train_loader, optimizer=optimizer,
            epoch=epoch, log_interval=log_interval, tr_loss=train_loss_tmp, verbose=verbose)
        
        print('EPOCHE: ', epoch, '=== Elapsed time: ', time.time() - start_time, 'seconds')

        for i in range(len(train_loss_tmp)):
            tr_loss_step.append(train_loss_tmp[i])
        tr_loss.append(np.sum(train_loss_tmp).item() / len(train_loader.dataset))
        if verbose == True:
            print(f'Train Epoch: {epoch} \t Average: {tr_loss[-1]:.8f}')

        scheduler.step(test_loss[-1])

    training_loss = []
    if len(tr_loss) == len(test_loss):
        for i, train_loss in enumerate(tr_loss):
            training_loss.append([train_loss, test_loss[i]])

    arr2cvs_file('loss.cvs', training_loss, 2, 'Training Loss', 'Test Loss')

    plot_training(tr_loss_step, tr_loss, test_loss, epochs, train_loader, batch_size)
    validate(model=VAE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=True, save_plot_fullpath=root_directory+'generated.png', verbose=verbose)

    torch.save(VAE_model.state_dict(), root_directory + 'pytorch_model_weights.pt')


# =========================== VANILLA === LATENT SPACE VISUALIZATION =========================================
if model_type == 'AE' and check_health == False:
    AE_model.eval()

    # Calculate Latents out of test dataset
    latents = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda: 
                data = data.cuda()
            _, _, z = AE_model(data)
            latents.append(z.detach().cpu())
            labels.append(target)

    latents = torch.cat(latents).numpy()
    labels = torch.cat(labels).numpy()

    # Visualize these latents in a simple plot
    plt.figure(figsize=(5, 5))
    for i in range(10):
        idxs = labels == i
        plt.scatter(latents[idxs, 0], latents[idxs, 1], label=str(i), alpha=0.5)

    plt.legend()
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.grid(True)
    plt.savefig(root_directory+'latent.png', format='png', dpi=420)
    plt.show()


# =========================== VARIATIONAL === LATENT SPACE VISUALIZATION =========================================
if model_type == 'VAE' and check_health == False:
    VAE_model.eval()

    # Calculate Latents out of test dataset
    latents = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda: 
                data = data.cuda()
            _, _, z, _ = VAE_model(data)
            latents.append(z.detach().cpu())
            labels.append(target)

    latents = torch.cat(latents).numpy()
    labels = torch.cat(labels).numpy()

    # Visualize these latents in a simple plot
    plt.figure(figsize=(5, 5))
    for i in range(10):
        idxs = labels == i
        plt.scatter(latents[idxs, 0], latents[idxs, 1], label=str(i), alpha=0.5)

    plt.legend()
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.grid(True)
    plt.savefig(root_directory+'latent.png', format='png', dpi=420)
    plt.show()




end_time = time.time()
print(" --- END OF CODE! --- %s seconds ---" % (end_time - start_time))
