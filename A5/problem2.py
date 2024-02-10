import torch.nn as nn


class CNN(nn.Module):
    """
    Constructs a convolutional neural network according to the architecture in the exercise sheet using the layers in torch.nn.

    Args:
        num_classes: Integer stating the number of classes to be classified.
    """
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        # Convolutional Layer
        self.conv = nn.Sequential(
            # layer 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Full connection layer
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class LossMeter(object):
    """
    Constructs a loss running meter containing the methods reset, update and get_score. 
    Reset sets the loss and step to 0. Update adds up the loss of the current batch and updates the step.
    get_score returns the runnung loss.

    """

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0.
        self.step = 0.

    def update(self, loss):
        self.loss += loss
        self.step += 1

    def get_score(self):
        return self.loss / self.step


def analysis():
    """
    Compare the performance of the two networks in Problem 1 and Problem 2 and
    briefly summarize which one performs better and why.
    """
    alys = """
    CNN Network performs better: obviously we can get accuracy 80% (in train set) by NN with 20 epochs, while CNN 
    already get accuracy 100% from 6th epoch.
    
    we thought because in convolutional layers each kernel uses the same weights when prosessing inputs, this will 
    reduce the number of parameters. Otherwise because the sparse connectivity in convolutional layers (the 
    convolutional kernels only connected to local regions of the input, will not be influenced by other regions) 
    further reducing computational costs.
    """
    print(alys)
