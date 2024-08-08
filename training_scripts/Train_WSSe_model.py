import torch
from GombNet.utils import *
from GombNet.networks import *
from GombNet.loss_func import GombinatorialLoss

# Create dataloaders
images_dir='/Users/austin/Desktop/WSSe_dataset/images'
labels_dir='/Users/austin/Desktop/WSSe_dataset/labels'
train_loader, val_loader, test_loader = get_dataloaders(images_dir, labels_dir, batch_size = 2, val_split=0.2, test_split=0.1)

save_name = '/Users/austin/Documents/GitHub/GombNet/trained_models/del_later.pth'

# Model Params
input_channels = 1
num_classes = 6    # number of output classes
num_filters = [32, 64, 128, 256, 512]

# Check if a GPU is available and if not, use a CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps') # backend for Apple silicon GPUs
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Create and train model with CombinatorialGroupLoss
print('Loading model...')
model = TwoLeggedUnet(input_channels, num_classes, num_filters, dropout = 0.423)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = GombinatorialLoss(group_size = num_classes//2, loss = 'Dice', epsilon=1e-6, class_weights = None, alpha=1.58)
print('Training model...')                         
model, train_loss, val_loss = train_model(model, train_loader, val_loader, n_epochs = 30,
                                          criterion = loss, optimizer = optimizer, device = device,
                                          save_name = save_name)