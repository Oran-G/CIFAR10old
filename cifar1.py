from cifar_train import train
from datetime import datetime
import os
import torch
import random
# define
batch, lr, epochs, bp = 128, 0.00075, 128, 250
# train
out = train(epochs=epochs, learning_rate=lr, batch_per= bp, batch=batch)
# out[0].savefig(f'./{folder}/loss.png')
PATH = f'.cifar_net.pth'
torch.save(out[1].state_dict(), PATH)
