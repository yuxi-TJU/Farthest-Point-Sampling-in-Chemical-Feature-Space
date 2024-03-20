from torch import nn

class FCLayer(nn.Module):
    def __init__(self, input_size, output_size, activation='gelu'):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')
        
    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out

class ANN(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size = 32, 
        num_hidden_layers = 1,
        output_size = 1,
        activation = 'gelu'
        ):
        super(ANN, self).__init__()
        module_list = []
        for i in range(num_hidden_layers):
            if i == 0:
                module_list.append(FCLayer(input_size, hidden_size, activation))
            else:
                module_list.append(FCLayer(hidden_size, hidden_size, activation))
        self.fc = nn.Sequential(*module_list)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc(x)
        out = self.out(out)
        return out