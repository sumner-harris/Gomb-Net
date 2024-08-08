import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha  # Weight for penalizing false positives (alpha=1 is neutral, alpha>1 penalizes false positives, alpha<1 penalizes false negatives)

    def forward(self, input, target):
        input = input.view(-1)  # Flatten the input
        target = target.view(-1)  # Flatten the target
        intersection = torch.sum(input * target)
        # Apply a weight to the prediction sum to penalize false positives
        weighted_union = self.alpha * torch.sum(input) + torch.sum(target)
        dice_coeff = (2. * intersection + self.smooth) / (weighted_union + self.smooth)
        dice_coeff = torch.clamp(dice_coeff, 0, 1)
        return 1. - dice_coeff

class GombinatorialLoss(nn.Module):
    def __init__(self, group_size, loss='Dice', epsilon=1e-6, class_weights=None, alpha=2, sim_penalty=True):
        super(GombinatorialLoss, self).__init__()
        self.group_size = group_size
        self.epsilon = epsilon
        self.class_weights = class_weights
        self.loss = loss.lower()
        self.alpha = alpha # for Dice loss
        self.sim_penalty = sim_penalty

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        total_loss = 0.0

        # Apply sigmoid to outputs if using Dice loss
        if self.loss == 'dice':
            outputs = torch.sigmoid(outputs)

        dice_loss = DiceLoss(alpha=self.alpha) if self.loss == 'dice' else None

        for i in range(batch_size):
            outputs_group1, outputs_group2 = outputs[i, :self.group_size], outputs[i, self.group_size:]
            targets_group1, targets_group2 = targets[i, :self.group_size], targets[i, self.group_size:]

            if self.loss == 'dice':
                loss00 = dice_loss(outputs_group1, targets_group1)
                loss01 = dice_loss(outputs_group1, targets_group2)
                loss10 = dice_loss(outputs_group2, targets_group1)
                loss11 = dice_loss(outputs_group2, targets_group2)
                output_loss = dice_loss(outputs_group1, outputs_group2) # maybe the penatly should be different
                target_loss = dice_loss(targets_group1, targets_group2) # for these two (modify alhpha)
            else:
                loss00 = F.cross_entropy(outputs_group1.unsqueeze(0), targets_group1, weight=self.class_weights, reduction='mean')
                loss01 = F.cross_entropy(outputs_group1.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')
                loss10 = F.cross_entropy(outputs_group2.unsqueeze(0), targets_group1, weight=self.class_weights, reduction='mean')
                loss11 = F.cross_entropy(outputs_group2.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')
                output_loss = F.cross_entropy(outputs_group1.unsqueeze(0), outputs_group2, weight=self.class_weights, reduction='mean')
                target_loss = F.cross_entropy(targets_group1.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')

            # Compute the inverse of loss pairings and sum for current sample
            inverse_loss = 1 / (loss01 + loss10 + self.epsilon) + 1 / (loss00 + loss11 + self.epsilon)
            prediction_loss = 1 / (inverse_loss + self.epsilon)

            if self.sim_penalty:
                # Loss penalizing similar predictions for G1 and G2
                mse_loss = (target_loss - output_loss) ** 2
                mse_loss = torch.sigmoid(mse_loss)

                # Accumulate the loss
                total_loss += prediction_loss * mse_loss
            else:
                # Accumulate the loss
                total_loss += prediction_loss

        # Average the accumulated losses over the batch
        return total_loss / batch_size

