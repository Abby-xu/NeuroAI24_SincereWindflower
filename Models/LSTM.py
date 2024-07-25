import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 

# LSTM with fully connected layer 
class LSTM_FC(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(LSTM_FC, self).__init__()
        self.output_size = output_size #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, output_size) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out, hn, cn #returns output and hidden state

# "Clasic" LSTM (i.e., without fully connected layer)
class LSTM_Vanilla (nn.Module):
    def __init__(self,output_size, input_size, latent_size, num_layers=1):
        super().__init__()
        self.output_size = output_size #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.latent_size = latent_size #hidden state
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=latent_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(latent_size, output_size)

    def forward(self, x, hidden):
        print(f'the shape of x is: {x.shape}')
        h_0 = Variable(torch.zeros(self.num_layers,x.size(0), self.latent_size)) #hidden state
        print(h_0.shape)
        c_0 = Variable(torch.zeros(self.num_layers,x.size(0), self.latent_size)) #internal state
        x, (hn, cn) = self.lstm(x,(h_0, c_0))
        x = self.linear(x)
        return x, hn
    
class LSTM_cell(nn.Module):
    def __init__(self, output_size, input_size, latent_size, num_layers=1):
        super(LSTM_cell, self).__init__()
        # Model parameters
        self.output_size = output_size #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.latent_size = latent_size #hidden state
        # Model layers
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=latent_size)
        self.linear = nn.Linear(latent_size, output_size)

    def init_hidden(self, batch_size):
        # Latent state initialization
        hx = torch.zeros(batch_size, self.latent_size)
        cx = torch.zeros(batch_size, self.latent_size)
        return (hx, cx)
        
    def forward(self, x, hidden):
        hx,cx = hidden
        print(f'the shape of hx is: {hx.shape}')
        print(f'hidden is a tuple of length: {len(hidden)}')
        hx, cx = self.lstm(x, (hx, cx))
        output = self.linear(hx)
        return output, (hx, cx)
    
class LSTM_Methyl(nn.Module):
    def __init__(
        self, latent_size, input_size=None, output_size=None
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = nn.LSTMCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = nn.LSTMCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def init_hidden(self, batch_size):
        H = torch.zeros(batch_size,self.latent_size)
        C = torch.zeros(batch_size,self.latent_size)
        return (H,C)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        h,c = hidden
        output = self.readout(h)
        return output, hidden
    

    
