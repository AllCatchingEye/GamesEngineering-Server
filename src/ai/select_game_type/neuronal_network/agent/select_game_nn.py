from torch import nn

# depending on the shape of the trainings-data `data.csv` and
# the result of `output_data.astype("category").cat.codes` (see also `train_classifier.ipynb` > Preprocessing)
input_size = 32
output_size = 18


class SelectGameNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
