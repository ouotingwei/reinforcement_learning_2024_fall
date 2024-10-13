import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNetDQN(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDQN, self).__init__()
        # Convolution layers
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True)
        )

        # Value stream (for estimating the value of the state)
        self.value_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)  # Output one scalar: V(s)
        )

        # Advantage stream (for estimating advantages of actions)
        self.advantage_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)  # Output: A(s, a) for each action
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.0  # Normalize the input
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)

        # Compute value and advantage separately
        value = self.value_stream(x)  # Shape: [batch_size, 1]
        advantage = self.advantage_stream(x)  # Shape: [batch_size, num_classes]

        # Combine value and advantage to compute Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AtariNetDQN(nn.Module):
#     def __init__(self, num_classes=4, init_weights=True):
#         super(AtariNetDQN, self).__init__()
#         self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
#                                         nn.ReLU(True),
#                                         nn.Conv2d(32, 64, kernel_size=4, stride=2),
#                                         nn.ReLU(True),
#                                         nn.Conv2d(64, 64, kernel_size=3, stride=1),
#                                         nn.ReLU(True)
#                                         )
#         self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
#                                         nn.ReLU(True),
#                                         nn.Linear(512, num_classes)
#                                         )

#         if init_weights:
#             self._initialize_weights()

#     def forward(self, x):
#         x = x.float() / 255.
#         x = self.cnn(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0.0)