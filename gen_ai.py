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

# If the model should only run the health checks and not train
check_health = False
verbose = False
plot_progress = True

epoches = 3
learning_rate = 0.01

#log_interval * batchsize = how often the intermediate result is displayed
batch_size = 300
log_interval = 10

test_batch_size = 1000

# LeakyReLU leak value
leak = 0
# ========================= PARAMETERS ==============================

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
print('Epoches: ', epoches)
print('Batch Size: ', batch_size)
print('Learning Rate: ', learning_rate)
print('ReLU Leak: ', leak)
print('Verbose: ', verbose)
print('Only Check Health: ', check_health)

# ================== HELPER FUNCTIONS ==========================================
#takes an array as the input and saves it at the specified path as a cvs file
def arr2cvs_file(path: str, array, width: int, *label):  
    '''
    Takes an 2D array as an input and saves it to a cvs file.
    The shape must be consistent. And the width specified.
    '''
    try:
        with open(path, 'ax') as file:
            heading = str(label[0])
            for i in range(width-1):
                heading = heading + ',' + str(label[i+1])

            file.write(heading + '\n')
            for i, item in enumerate(array):

                entry = str(item[0])

                for y in range(width-1):
                    print(entry)
                    entry = entry + ',' + str(item[y+1])
                print(entry)
                file.write( entry + '\n' )

    except:
        print("Error: Something went wrong")
    pass



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
                 **kwargs) -> None:
    
        super(Vanilla_AE, self).__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.leak = leak

        modules = []
        if hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256]
        else: 
            self.hidden_dims = hidden_dims

        # TODO: Build Encoder
        # Tipp: You can use a loop over the hidden dims in combination with a nn.Sequential


        # TODO: Save complete Encoder Sequential as self.encoder
        layers = self.hidden_dims


        self.encoder = nn.Sequential(
            nn.Conv2d(1, layers[0], 3, 2, 1),
            nn.BatchNorm2d(layers[0]),
            nn.LeakyReLU(self.leak),

            nn.Conv2d(layers[0], layers[1], 3, 2, 1),
            nn.BatchNorm2d(layers[1]),
            nn.LeakyReLU(self.leak),

            nn.Conv2d(layers[1], layers[2], 3, 2, 1),
            nn.BatchNorm2d(layers[2]),
            nn.LeakyReLU(self.leak),

            nn.Conv2d(layers[2], layers[3], 3, 2, 1),
            nn.BatchNorm2d(layers[3]),
            nn.LeakyReLU(self.leak)
        )

        # We basically calculate the image resolution here 
        # Since we use a fixed latent dim we need to calculate what the latent dim will be with our 
        # fixed image size. Since we always use stride=2, with each block we reduce the image resolution
        # by 2. 
        # Because // always rounds down, while nn.Conv2D will round up we add 2 - 1 to get the rounded up 
        # result.
        self.block_dimension = self.input_size 
        #print(self.block_dimension)
        for i in range(len(self.hidden_dims)): 
            self.block_dimension = (self.block_dimension + 2 - 1) // 2

        # This is used for the VAE later on 
        # For the AE we just use the fully_connecte mu predictor (fc_mu)
        #print(type(self.block_dimension))
        #print(self.block_dimension)
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * self.block_dimension[0] * self.block_dimension[1], latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * self.block_dimension[0] * self.block_dimension[1], latent_dim)


        # Build Decoder
        # Since the upsample results in images of 32 by 32 we need a resize operation in the final_layer
        # Remember to do this
        # TODO: Implement your decoder
        # You can do this similar to the encoder


        # TODO: Save complete Decoder Sequential as self.Decoder
        layers = self.hidden_dims
        self.decoder_ff = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dims[-1] * self.block_dimension[0] * self.block_dimension[1]),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(layers[3], layers[2], 3, 2, 1, 1),
            nn.BatchNorm2d(layers[2]),
            nn.LeakyReLU(self.leak),

            nn.ConvTranspose2d(layers[2], layers[1], 3, 2, 1, 1),
            nn.BatchNorm2d(layers[1]),
            nn.LeakyReLU(self.leak),

            nn.ConvTranspose2d(layers[1], layers[0], 3, 2, 1, 1),
            nn.BatchNorm2d(layers[0]),
            nn.LeakyReLU(self.leak),

            nn.ConvTranspose2d(layers[0], layers[0], 3, 2, 1, 1),
            nn.BatchNorm2d(layers[0]),
            nn.LeakyReLU(self.leak),
        )

        # TODO: Set up Final Layer of Transposed Convolution and Resizing
        self.final_layer = nn.Sequential(
            TT.Resize(28),
            nn.Conv2d(layers[0], 1, 1, 1, 0),
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
        encoding = self.encoder(input)
        encoding = encoding.flatten(1)
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
        
        z = self.decoder_ff(z)
        z = torch.reshape(z, (-1, 256, 2, 2))
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
        loss = nn.functional.mse_loss(prediction[0], target)
        return loss


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
                 **kwargs) -> None:
        super().__init__(in_channels, latent_dim, input_size, hidden_dims)

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

        kl_loss = 0.5 * torch.sum( mu * mu + sigma - log_sigma - 1) / num_batches 
        mse_loss = 0.5 * torch.sum( torch.pow(input - reconstruction, 2), 1)

        loss = kl_loss + mse_loss.mean()

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
    AE_model = Vanilla_AE(in_channels=1, latent_dim=2, input_size=np.array([28, 28]), hidden_dims=[32, 64, 128, 256])

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
    VAE_model = VAE(in_channels=1, latent_dim=2, input_size=np.array([28, 28]), hidden_dims=[32, 64, 128, 256])

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
        if epoch % 3 == 0:
            validate(model=AE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=plot_progress, verbose=verbose)
        else:
            validate(model=AE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=False, verbose=verbose)
        # train one epoch
        train_loss_tmp = []
        train(model=AE_model, use_cuda=use_cuda, train_loader=train_loader, optimizer=optimizer,
            epoch=epoch, log_interval=log_interval, tr_loss=train_loss_tmp, verbose=verbose)
        for i in range(len(train_loss_tmp)):
            tr_loss_step.append(train_loss_tmp[i])
        tr_loss.append(np.sum(train_loss_tmp).item() / len(train_loader.dataset))
        if verbose == True:
            print(f'Train Epoch: {epoch} \t Average: {tr_loss[-1]:.8f}')

        scheduler.step(test_loss[-1])

    plot_training(tr_loss_step, tr_loss, test_loss, epochs, train_loader, batch_size)


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
        if epoch % 5 == 0:
            validate(model=VAE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=plot_progress, verbose=verbose)
        else:
            validate(model=VAE_model, use_cuda=use_cuda, test_loader=test_loader, test_loss=test_loss, plot=False, verbose=verbose)
        # train one epoch
        train_loss_tmp = []
        train(model=VAE_model, use_cuda=use_cuda, train_loader=train_loader, optimizer=optimizer,
            epoch=epoch, log_interval=log_interval, tr_loss=train_loss_tmp, verbose=verbose)
        
        print('EPOCHE: ', epoch)

        for i in range(len(train_loss_tmp)):
            tr_loss_step.append(train_loss_tmp[i])
        tr_loss.append(np.sum(train_loss_tmp).item() / len(train_loader.dataset))
        if verbose == True:
            print(f'Train Epoch: {epoch} \t Average: {tr_loss[-1]:.8f}')

        scheduler.step(test_loss[-1])

    #if len(tr_loss) == len(test_loss):
    #    for i, train_loss in enumerate(tr_loss):
    #        print(train_loss, test_loss[i])
    #arr2cvs_file(path_data_loss, [])

    plot_training(tr_loss_step, tr_loss, test_loss, epochs, train_loader, batch_size)



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
    plt.show()





print(" END OF CODE! ")