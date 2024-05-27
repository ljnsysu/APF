import torch.nn as nn
import torch


class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator.
    Args:
        image_size (int): The size of the image. (Default: 28)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 10)
    """

    def __init__(self, in_dim: int = 32, out_dim: int = 32, num_classes: int = 13) -> None:
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, self.out_dim),
            # nn.Dropout(0.5)
            # nn.ReLU(inplace=True)
            # nn.Linear(self.in_dim, 128),
            # nn.ReLU(inplace=True),
            #
            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),
            #
            # nn.Linear(256, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            #
            # nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(inplace=True),
            #
            # nn.Linear(1024, self.out_dim),
            # nn.Tanh()
            # nn.ReLU(inplace=True)
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.
        Returns:
            A four-dimensional vector (N*C*H*W).
        """
        # print(inputs.shape)
        # print(self.label_embedding(labels.long()).shape)
        # conditional_inputs = torch.cat([inputs, labels], dim=-1)
        # out = self.main(conditional_inputs)
        out = self.main(inputs)
        # out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
# class Generator(nn.Module):
#     def __init__(self, in_dim=32, out_dim=32):
#         super(Generator, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.main = nn.Sequential(
#             nn.Conv2d(self.in_dim, self.out_dim, 1, 1, 0, bias=True),
#             nn.BatchNorm2d(self.out_dim),
#             nn.ReLU(True),
#
#             nn.Conv2d(self.out_dim, self.out_dim * 2, 1, 1, 0, bias=True),
#             nn.BatchNorm2d(self.out_dim * 2),
#             nn.ReLU(True),
#
#             nn.Conv2d(self.out_dim * 2, self.out_dim * 4, 1, 1, 0, bias=True),
#             nn.BatchNorm2d(self.out_dim * 4),
#             nn.ReLU(True),
#
#             nn.Conv2d(self.out_dim * 4, self.out_dim * 2, 1, 1, 0, bias=True),
#             nn.BatchNorm2d(self.out_dim * 2),
#             nn.ReLU(True),
#
#             nn.Conv2d(self.out_dim * 2, self.out_dim, 1, 1, 0, bias=True),
#         )
#
#     def forward(self, input):
#         return self.main(input)