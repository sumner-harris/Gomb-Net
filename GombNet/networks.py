import torch
import torch.nn as nn
from GombNet.loss_func import DiceLoss
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv_block(x)
        x_pooled = self.pool(conv)
        return conv, x_pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


# Classic Unet
# -------------------------------------
class Unet(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, dropout = 0.1):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # Encoder
        for i, filters in enumerate(num_filters[:-1]):
            self.encoders.append(EncoderBlock(input_channels if i == 0 else num_filters[i-1], filters))

        # Bottleneck
        self.bottleneck = ConvBlock(num_filters[-2], num_filters[-1])

        # Decoder
        num_filters_reversed = num_filters[::-1]
        for i in range(len(num_filters) - 1):
            in_channels = num_filters_reversed[i]
            out_channels = num_filters_reversed[i + 1]  # This is the output channel size after ConvBlock
            self.decoders.append(DecoderBlock(in_channels, out_channels))

        # Classifier
        self.final_conv = nn.Conv2d(num_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, x_pooled = encoder(x)
            skips.append(x)
            x = x_pooled

        x = self.bottleneck(x)
        x = self.dropout(x)

        # reverse the skips list
        skips_reverse = skips[::-1]

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips_reverse[i])

        x = self.final_conv(x)

        return x


# Unet with Two Decoders
# -------------------------------------
class TwoLeggedUnet(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, dropout=0.1):
        super(TwoLeggedUnet, self).__init__()
        self.encoders = nn.ModuleDict()
        self.decoders1 = nn.ModuleDict()
        self.decoders2 = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        # Encoder
        for i, filters in enumerate(num_filters[:-1]):
            self.encoders[f'encoder_{i}'] = EncoderBlock(input_channels if i == 0 else num_filters[i-1], filters)

        # Bottleneck
        self.bottleneck = ConvBlock(num_filters[-2], num_filters[-1])

        # Decoder 1
        num_filters_reversed = num_filters[::-1]
        for i in range(len(num_filters) - 1):
            in_channels = num_filters_reversed[i]
            out_channels = num_filters_reversed[i + 1]
            self.decoders1[f'decoder1_{i}'] = DecoderBlock(in_channels, out_channels)

        # Decoder 2
        for i in range(len(num_filters) - 1):
            in_channels = num_filters_reversed[i]
            out_channels = num_filters_reversed[i + 1]
            self.decoders2[f'decoder2_{i}'] = DecoderBlock(in_channels, out_channels)

        # Classifiers
        self.final_conv1 = nn.Conv2d(num_filters[0], num_classes // 2, kernel_size=1)
        self.final_conv2 = nn.Conv2d(num_filters[0], num_classes // 2, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders.values()):
            x, x_pooled = encoder(x)
            skips.append(x)
            x = x_pooled

        x = self.bottleneck(x)
        x = self.dropout(x)

        # reverse the skips list
        skips_reverse = skips[::-1]

        # Decoder 1
        x1 = x
        for i, decoder in enumerate(self.decoders1.values()):
            x1 = decoder(x1, skips_reverse[i])
        x1 = self.final_conv1(x1)

        # Decoder 2
        x2 = x
        for i, decoder in enumerate(self.decoders2.values()):
            x2 = decoder(x2, skips_reverse[i])
        x2 = self.final_conv2(x2)

        x_concat = torch.cat([x1, x2], dim=1)
        
        return x_concat
    
# Ensemble class for voting on outputs
# -------------------------------------
class GombSemble:
    def __init__(self, models):
        """
        Initialize the GombSemble class with models dict
        """
        self.models = models  # Dictionary or list of models in the ensemble
        self.image = None

        self.outputs = None
        self.predictions = None
        self.ordered_outputs = None


    def predict(self, image, return_plot=True, mode='threshold',stdvs_n=2):
        """
        Run predictions using the ensemble of models and store the outputs.
        """
        self.image = image
        self.outputs = []
        self.predictions = []
        for mod in self.models:
            # Select model
            model = self.models[mod]
            print(f"Running model: {mod}")

            # Make predictions
            with torch.no_grad():
                outputs = model(self.image).squeeze(0)  # Assume a batch size of 1
                self.outputs.append(outputs)
            
            # Make binary predictions
            prediction = self.handle_outputs(outputs, mode=mode, stdvs_n=stdvs_n)
            self.predictions.append(prediction)

            # Plot the predictions
            if return_plot:
                self.plot_predictions(outputs, prediction)


    def handle_outputs(self, outputs, mode='threshold', stdvs_n=3):
        """
        Handle outputs for thresholding or other post-processing.
        This function can be customized further based on what 'handle_outputs' should do.
        """
        # Example handling function, this can be expanded based on specific needs
        return (outputs > stdvs_n).float() if mode == 'threshold' else outputs
    

    def rearrange_ensemble(self):
        """
        Rearrange the ensemble outputs based on the Dice loss.
        """
        # Use the first model's group as target for reordering others
        rearranged_ensemble = []
        group1 = [output[0:3, :, :] for output in self.outputs]
        group2 = [output[3:6, :, :] for output in self.outputs]
        
        target_group1 = group1[0]
        target_group2 = group2[0]
        
        rearranged_group1 = [target_group1]
        rearranged_group2 = [target_group2]

        for model in range(1, len(self.outputs)):
            current_group1 = group1[model]
            current_group2 = group2[model]

            self.dice_loss_fn = DiceLoss()  # Initialize the Dice loss function from loss module
            # Compute losses for both swap and non-swap scenarios
            loss_no_swap = (self.dice_loss_fn(target_group1, current_group1).item() +
                            self.dice_loss_fn(target_group2, current_group2).item()) / 2.0
            loss_swap = (self.dice_loss_fn(target_group1, current_group2).item() +
                         self.dice_loss_fn(target_group2, current_group1).item()) / 2.0

            if loss_swap < loss_no_swap:
                rearranged_group1.append(current_group2)
                rearranged_group2.append(current_group1)
            else:
                rearranged_group1.append(current_group1)
                rearranged_group2.append(current_group2)

        rearranged_ensemble = [torch.cat([rearranged_group1[i], rearranged_group2[i]], dim=0) 
                               for i in range(len(rearranged_group1))]
        self.ordered_outputs = rearranged_ensemble


    def plot_predictions(self, outputs, prediction):
        """
        Plot the predictions and the outputs.
        """
        num_classes = outputs.shape[0]
        fig, axs = plt.subplots(2, num_classes, sharex=True, sharey=True, figsize=(15, 5))

        for i in range(num_classes):
            axs[0, i].imshow(prediction[i], cmap='gray')
        for i in range(num_classes)[:2]:
            axs[1, i].imshow(outputs[i], cmap='plasma')
        for i in range(num_classes)[2:]:
            axs[1, i].imshow(outputs[i], cmap='viridis')
        for ax in axs.ravel():
            ax.axis('off')
        axs[0, 0].set_ylabel('Prediction')
        axs[1, 0].set_ylabel('Probability')

        fig.tight_layout()
        plt.show()


    def vote(self, mode='mean', std_blur=1):
        """
        Vote on the predictions using the ensemble of models.
        """
        rearranged_ensemble = self.ordered_outputs # simplify this line later
        rearranged_np = np.array([tensor.numpy() for tensor in rearranged_ensemble])  # Convert to numpy array
        self.voted_stddev = np.std(rearranged_np, axis=0)
        self.voted_stddev = gaussian_filter(self.voted_stddev, std_blur)

        # Perform voting
        if mode == 'mean':
            average_prediction = np.mean(rearranged_np, axis=0)
            self.voted_prediction = average_prediction

        elif mode == 'max':
            max_prediction = np.max(rearranged_np, axis=0)
            self.voted_prediction = max_prediction


    def plot_vote(self):
        """
        Plot the average and standard deviation across the rearranged ensemble.
        """

        num_channels = self.voted_prediction.shape[0]  # Should be 6

        fig, ax = plt.subplots(2, num_channels, figsize=(15, 10))
        for channel in range(num_channels):
            # Plot average prediction
            ax[0,channel].set_title(f'Channel {channel} mean')
            ax[0,channel].imshow(self.voted_prediction[channel], cmap='viridis')
            ax[0,channel].axis('off')

            # Plot standard deviation
            ax[1,channel].set_title(f'Channel {channel} stddev')
            ax[1,channel].imshow(self.voted_stddev[channel], cmap='magma')
            ax[1,channel].axis('off')

        ax[0,0].set_ylabel('Average Prediction')
        ax[1,0].set_ylabel('Standard Deviation')
        fig.tight_layout()


class OriginalUnet(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters):
        super(OriginalUnet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for i, filters in enumerate(num_filters[:-1]):
            self.encoders.append(EncoderBlock(input_channels if i == 0 else num_filters[i-1], filters))

        # Bottleneck
        self.bottleneck = ConvBlock(num_filters[-2], num_filters[-1])

        # Decoder
        num_filters_reversed = num_filters[::-1]
        for i in range(len(num_filters) - 1):
            in_channels = num_filters_reversed[i]
            out_channels = num_filters_reversed[i + 1]  # This is the output channel size after ConvBlock
            self.decoders.append(DecoderBlock(in_channels, out_channels))

        # Classifier
        self.final_conv = nn.Conv2d(num_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, x_pooled = encoder(x)
            skips.append(x)
            x = x_pooled

        x = self.bottleneck(x)

        # reverse the skips list
        skips_reverse = skips[::-1]

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips_reverse[i])

        x = self.final_conv(x)

        return x

