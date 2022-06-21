import torch
import torch.nn.functional as F
from torch.autograd import Variable

input_train = torch.tensor([[80., -80., 80., -80.],
                            [80., -80., 80., -128.],
                            [80., -128., 128., -80.],
                            [128., -80., 80., -128.],
                            [-80., 80., -80., 80.],
                            [-80., 128., -80., 128.],
                            [-128., 80., -128., 80.],
                            [-128., 80., -128., 128.]])

output_train = torch.tensor([[0.20384613, 0.72682253, 0.74039651, 0.4553828],
                             [0.02952115, 0.67100917, 0.64511755, 0.41593989],
                             [0.13867507, 0.24298393, 0.69337157, 0.0276931],
                             [0.15465341, 0.67162926, 0.61800869, 0.49935838],
                             [0.11135052, 0.56186498, 0.69144333, 0.25258834],
                             [0.        , 0.50749285, 0.46968882, 0.23991102],
                             [0.13217781, 0.78881119, 0.55447281, 0.00297377],
                             [1.66338846e-04, 7.23459942e-01, 3.56036130e-01, 2.91157868e-02]])

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.elu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

def SaveNNModel(model, save_path):
    torch.save(model, save_path)

net = Net(n_feature=4,n_hidden=16,n_output=4)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=.001)
loss_func = torch.nn.MSELoss()

for i in range(5000):
    x = Variable(input_train)
    y = Variable(output_train)

    prediction = net(x)
    loss = loss_func(prediction,y)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()

for i in range(8):
    t1 = Variable(input_train[i,:])
    print('Test Data: ',t1)
    a = Variable(output_train[i,:])
    print('Analytic Result: ',a)
    prediction = net(t1)
    print('NN prediction: ',prediction)
    print()

SaveNNModel(net,"../curve_following/friction_NN_model/friction_NN")
