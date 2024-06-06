# Source: https://github.com/StanfordASL/scod-module/blob/main/demos/1d_regression.ipynb

%load_ext autoreload
%autoreload 2

import scod
import torch
import numpy as np
from tqdm import trange

from matplotlib import pyplot as plt


# THIS SCRIPT IS NOT MEANT TO BE RUN AS IS! - PLEASE MAKE ADJUSTMENTS BELOW TO MAKE IT COMPATIBLE FOR OUR CASE BY COMPLETING "TODO" TAGS


## Load dist layer - I don't know which one to choose yet
dist_layer = # TODO: Select dist layer
# dist_layer = scod.distributions.NormalMeanParamLayer() 
# OR dist_layer = scod.distributions.NormalMeanDiagVarParamLayer()

## Attach dist layer to model and train it with our dataset
model = # TODO: Load our trained model and call it "model"
joint_model = torch.nn.Sequential(model, dist_layer) 

dataset = # TODO: use our dataset
# might need to experiment with different values for "batch_size" and "lr"
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# OPTIONAL - record "losses" for all epochs to plot training progress for model
losses = []

# "trange" object is used to create a progress bar to visualize training progress
t = trange(200)

# for 200 epochs, where 200 is the length of the trange object
for epoch in t:
    epoch_loss = 0.
    for (inputs, targets) in train_dataloader:

        # Clear the gradients from the previous step.
        optimizer.zero_grad()

        # Pass the inputs through the model to get the output distribution.
        dist = model(inputs)

        # Calculate the negative log-probability of the targets under the output distribution (this is done for probabilistic models)
        loss = -dist.log_prob(targets)

        # Compute the mean loss for the current batch.
        mean_loss = loss.mean()

        # Perform backpropagation to compute the gradients.
        mean_loss.backward()

        # Accumulate the mean loss for the current batch to the epoch loss.
        epoch_loss += mean_loss
        
        # Update the model parameters using the computed gradients.
        optimizer.step()
    
    # Compute the average loss for the epoch by dividing the accumulated loss by the number of batches.
    epoch_loss /= len(train_dataloader)
    
    # Set the progress bar description to display the average loss for the current epoch.
    t.set_description("mean_loss=%02f"%epoch_loss.item())

    # Append the average epoch loss to the losses list
    losses.append(epoch_loss.item())

# OPTIONAL - plot the losses
plt.plot(losses)

## 