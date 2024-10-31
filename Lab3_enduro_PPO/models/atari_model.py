import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AtariNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNet, self).__init__()
        # Convolution layers
        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.ReLU(True)
                                        )
        # actor-critic
        # output = num_classes's probability distribution
        self.action_logits = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        # output = the value of the state
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, 1)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False, a=[]):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)

        # get the state value
        value = self.value(x)
        value = torch.squeeze(value)

        # get num_classes's probability distribution
        logits = self.action_logits(x)
        
        dist = Categorical(logits=logits)
        
        ### TODO ###
        # Finish the forward function
        # evaluation -> choose the action which had the hightest probability 
        if eval:
            action = torch.argmax(logits, dim=1)
        
        # training -> random sample
        else:
            action = dist.sample()

        if len(a) == 0:
            log_probability = dist.log_prob(action)
        else:
            log_probability = dist.log_prob(a)

        log_probability = torch.squeeze(log_probability)
        entropy = dist.entropy().mean()

        return action, log_probability, value, entropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)