# -*- coding: utf-8 -*-
#import numpy as np
import torch
import torch.utils.data
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import *
from matplotlib.pyplot import *
from collections import OrderedDict
from scipy import stats, signal
import copy
from mpl_toolkits import mplot3d
import torch.nn.functional as F


mpl.style.use('seaborn')
cmap = matplotlib.cm.get_cmap('winter')
device = 'cuda:6' #CHANGE THIS TO 'cpu' if running locally
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10

H_tot = 200
numH = 0 #Number of Hidden Layers MINUS THE FIRST ONE
# H_2 = 10
N, D_in, H, D_out = 256, 1, 100 , 1
H,H_2 = int(H_tot/(numH+1)),int(H_tot/(numH+1))
NumIters = 10000
PlotFreq = 100
PrintFreq = 1000

MinInput = -3.
MaxInput = +3.
NoiseStdDev = 0

learning_rate = 2.0e-5

#1e-5 works for 500 hidden units, but not for 5000 hidden units
    #4e-05 is the highest that the learning rate can go.
    #At this rate, easily stuck in local minimum.

#4.28125e-6 is the maximum learning rate for 5000 hidden units
    #Actually performs quite poorly compared to 3e-06


#June 26, 2019:
# Need to figure out how to get the same initialization multiple times (probably just by setting a proper seed.)
#need to figure out how to save the mask.

plot_curve = False
plot_histograms = False
plot_residue = False
plot_loss = True
plot_deep_brk = False
mini_batch_option = False
freeze_v = False

initFunc = "Default"
activationFunc = "DeepReLu"

#Options for creating video files
# tag = 'PWGG/bad_init_pwgg_Adam'
tag = 'loss_plots/constantH1H2_scatter_v3'

getOut = True
makeVid = False       #Save the Animation File
savePic = True      #Save the final output image
freezeOrigin = False #Stop Training of First layer (w and b)
freezeY = False      #Stop Training of Second layer (v)
projectBrk = True   #Project Breakpoints onto axes
plotGrad = False     #Plot the gradients of the layers (w and v) !! To use must comment out other plots atm. !!
ims = []

#Options for Pruning
iter_prune = False
grow_prune = False
percent = .009
PruneFreq = int(NumIters/(16))
seed_num = 1111
seed = False
pruneBias = False #Only Matters if iter_prune = True

cmap4 = matplotlib.cm.get_cmap("hsv")

fig = figure()

# Create random Tensors to hold inputs and outputs
if seed:
    torch.manual_seed(seed_num)
x = torch.FloatTensor(N, D_in).uniform_(MinInput, MaxInput)
###creating a gap
# x = torch.FloatTensor(int(5/4 * N),D_in).uniform_(-3, 3)
# x = np.sort(x.numpy(),axis=0)
# x = torch.Tensor(np.concatenate((x[:int(N/4)],x[int(N/2):])))
###creating output function
noise = NoiseStdDev * torch.randn(N, D_out)
# y = torch.exp(x) + noise                        #exponential
y = torch.sin(np.pi*x) + noise                  #sine
# y = torch.Tensor((x-2.98)*(x)*(x+2.7))                 #cubic
# y = torch.Tensor((x-2.97)*(x-.32)*(x+1.47)*(x-2.5)*(x+2.92))   #5th order
# y= torch.pow(x,2) +noise                      #quadratic
# y =torch.pow(x-2,6)-2+noise                     #shifted quadratic
# y = x +noise                                    #linear
# y = torch.Tensor(signal.sawtooth(x.numpy())) #sawtooth
# y = torch.Tensor([np.sin(np.pi*val) if np.abs(val-.5)>1.5 else -1*np.cos(np.pi*(val-.5)/3) for val in x])
# y = x * torch.atan(x/2)
###use these in conjunction with another y
# y = 2.5*(y-1.1) + 2.1*(y+.35) - (y+.81) + 1.5*(y+.26) - 3.1*(y-.23)
# y = 2*y/max(np.abs(y))

#For the 2D outputNet.
# x1 = torch.FloatTensor(int(N/2)).uniform_(-3, -2)
# x2 = torch.FloatTensor(N-len(x1)).uniform_(2,3)
# x = torch.cat((x1,x2))
# y = torch.Tensor([[1,0] if val <0 else [0,1] for val in x])


x = x.to(device)
y = y.to(device)

breakPointsData = []
losses = []
iters = []
H_tot = 2000
for _ in range(5):
    for count,iter in enumerate([10,20,40,50,100,200,H_tot]):
        H = iter
        H_2 = 0 if H==H_tot else (H_tot/H)


        #Some helper functions

        def calculateBreakPoints(weight_tensor, bias_tensor):
            breakPointsLocations = np.divide(np.squeeze(bias_tensor), np.squeeze(weight_tensor))
            breakPointsLocations  = breakPointsLocations * -1.0
            temp = list(breakPointsLocations)
            plt.hist(temp, normed=True, bins=12, range=(-3, 3),edgecolor='black', linewidth=1.2)
            plt.ylabel('Distribution')


        def _calculate_fan_in_and_fan_out(tensor):
            dimensions = tensor.dim()
            if dimensions < 2:
                raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

            if dimensions == 2:  # Linear
                fan_in = tensor.size(1)
                fan_out = tensor.size(0)
            else:
                num_input_fmaps = tensor.size(1)
                num_output_fmaps = tensor.size(0)
                receptive_field_size = 1
                if tensor.dim() > 2:
                    receptive_field_size = tensor[0][0].numel()
                fan_in = num_input_fmaps * receptive_field_size
                fan_out = num_output_fmaps * receptive_field_size

            return fan_in, fan_out

        def _calculate_correct_fan(tensor, mode):
            mode = mode.lower()
            valid_modes = ['fan_in', 'fan_out']
            if mode not in valid_modes:
                raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

            fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
            return fan_in if mode == 'fan_in' else fan_out

        def calculate_gain(nonlinearity, param=None):

            linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
            if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
                return 1
            elif nonlinearity == 'tanh':
                return 5.0 / 3
            elif nonlinearity == 'relu':
                return math.sqrt(2.0)
            elif nonlinearity == 'leaky_relu':
                if param is None:
                    negative_slope = 0.01
                elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
                    # True/False are instances of int, hence check above
                    negative_slope = param
                else:
                    raise ValueError("negative_slope {} not a valid number".format(param))
                return math.sqrt(2.0 / (1 + negative_slope ** 2))
            else:
                raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))



        def prune_by_percent(percents, masks, final_weights):
          """Return new masks that involve pruning the smallest of the final weights.
          Args:
            percents: A dictionary determining the percent by which to prune each layer.
              Keys are layer names and values are floats between 0 and 1 (inclusive).
            masks: A dictionary containing the current masks. Keys are strings and
              values are numpy arrays with values in {0, 1}.
            final_weights: The weights at the end of the last training run. A
              dictionary whose keys are strings and whose values are numpy arrays.
          Returns:
            A dictionary containing the newly-pruned masks.
          """

          def prune_by_percent_once(percent, mask, final_weight):
            # Put the weights that aren't masked out in sorted order.
            sorted_weights = np.sort(np.abs(final_weight.data[torch.from_numpy(mask).view(final_weight.data.shape) == 1].cpu()))

            # Determine the cutoff for weights to be pruned.
            cutoff_index = np.round(percent * sorted_weights.size).astype(int)

            cutoff = sorted_weights[cutoff_index]

            # Prune all weights below the cutoff.
            return np.where(np.abs(final_weight.detach().cpu()) < cutoff, np.zeros(mask.shape), mask)

          new_masks = {}
          prod = torch.ones(H).view(1,-1).to(device)
          for k, percent in percents.items():
              if 'bias' in k:
                  if pruneBias:
                    new_masks[k] = prune_by_percent_once(percent,masks[k],final_weights[k])
                  else:
                    new_masks[k] = masks[k]
              else:
                  prod *= final_weights[k].view(1,-1)
          for k, percent in percents.items():
              if 'bias' not in k:
                  new_masks[k] = prune_by_percent_once(percent, masks[k], prod.view(final_weights[k].shape))
          return new_masks


        def pwgg(new_masks,x_vals,model_params,sd=2): #Function to implement PieceWise Gradient Generation
            #the new wi and vi are the same, also this literally only works with a 1 layer ReLu
            '''
            inputs:
            new_masks: Dictionary of masks of the neurons , with 0 indicating the units that will be changed
            x_vals: List of x_values that need a breakpoint at that location
            model_params: dict(model.named_parameters())
            sd: A scalar containing the stdev of the weights <- not in use atm
            return:
            gen_grad: values to add to all of the zeroed neurons
            '''

            # weights = []
            # biases = []
            # for val in x_vals:
            #     bias = np.random.randn() * sd
            #     weight = -1 * bias / val
            #     weights.append(weight)
            #     biases.append(bias)
            # for k,mask in new_masks.items():
            #     iter = 0
            #     mask = np.logical_xor(mask, 1)
            #     if "weight" in k:
            mask = new_masks['0.weight']
            mask = np.logical_xor(mask, 1).astype(float)
            iter = 0
            for idx in range(len(mask)):
                if mask[idx] == 1:
                    new_w =  (-1 * np.divide(model_params['0.bias'].data[idx].cpu(), x_vals[iter]))[0].numpy()
                    mask[idx][0] = new_w
                    iter += 1
            gen_grad = new_masks
            gen_grad['0.weight'] = mask
            gen_grad['2.weight'] = mask
            return gen_grad



        def weights_init(m):
            if isinstance(m, torch.nn.Linear):

                if initFunc == "Uniform":
                    stdv = 1. / math.sqrt(m.weight.data.size(1))
                    m.weight.data.uniform_(-stdv, stdv)
                    if m.bias is not None:
                        m.bias.data.uniform_(-stdv, stdv)

                elif initFunc == "BothUniform":
                    if (m.weight.data.size(0)) != 1:
                        m.weight.data.uniform_(MinInput, MaxInput)
                        if m.bias is not None:
                            m.bias.data.uniform_(MinInput, MaxInput)

                elif initFunc == "BreakPointUniformDistribution":
                    EvenlySpacedX = np.arange(MinInput,MaxInput,(MaxInput - MinInput)/H)
                    if (m.weight.data.size(0)) != 1:
                        # for i in range(m.weight.data.size(1)):
                        #     m.weight.data[0][i] = EvenlySpacedX[i]
                        stdv = 1. / math.sqrt(m.weight.data.size(1))
                        m.weight.data.uniform_(-stdv, stdv)
                        if m.bias is not None:
                            #m.bias.data.fill_(1.0)
                            m.bias.data.uniform_(-stdv, stdv)
                    else:
                        for i in range(m.weight.data.size(0)):
                            m.weight.data[i] = 1.0000 / EvenlySpacedX[i]
                        print(m.weight.data)
                        if m.bias is not None:
                            m.bias.data.fill_(-1.0)
                    m.weight.data = torch.nn.Parameter(m.weight.data)
                    m.bias.data = torch.nn.Parameter(m.bias.data)


                elif initFunc == "BreakPointUniformDistribution_v2":
                    if (m.weight.data.size(0)) != 1:
                        m.weight.data.fill_(1.0)
                        m.bias.data.uniform_(MinInput,MaxInput)
                        m.weight.data = torch.nn.Parameter(m.weight.data)
                        m.bias.data = torch.nn.Parameter(m.bias.data)


                elif initFunc == "BreakPointUniformDistribution_v3":
                    if (m.weight.data.size(0)) != 1:
                        EvenlySpacedX = np.arange(MinInput, MaxInput, (MaxInput - MinInput) / H)
                        m.weight.data.fill_(1.0)
                        for i in range(m.bias.data.size(0)):
                            m.bias.data[i] = EvenlySpacedX[i]
                        print (m.bias.data)
                        m.weight.data = torch.nn.Parameter(m.weight.data)
                        m.bias.data = torch.nn.Parameter(m.bias.data)

                elif initFunc == "BreakPointUniformDistribution_v4":
                    if (m.weight.data.size(0)) != 1:
                        m.weight.data.fill_(1.0)
                        m.bias.data.uniform_(MinInput,MaxInput)
                        m.weight.data = torch.nn.Parameter(m.weight.data)
                        m.bias.data = torch.nn.Parameter(m.bias.data)
                    else:
                        fan = _calculate_correct_fan(m.weight.data, "fan_in")
                        gain = calculate_gain('leaky_relu', math.sqrt(5))
                        std = gain / math.sqrt(fan)
                        bound = 4 * math.sqrt(3.0) * std
                        m.weight.data.uniform_(-bound,bound)

                        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_in)
                        torch.nn.init.uniform_(m.bias.data, -bound, bound)


                elif initFunc == "NormalDistributedBias":
                    if (m.weight.data.size(0)) != 1:
                        m.weight.data.fill_(1.0)
                        m.bias.data.normal_(0, 0.5)
                        m.weight.data = torch.nn.Parameter(m.weight.data)
                        m.bias.data = torch.nn.Parameter(m.bias.data)

                elif initFunc == "He":
                    torch.nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
                    if m.bias is not None:
                        #TODO june 21:
                        #Find weight and bias such that w/b gives a uniform distribution of
                        #break points.
                        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_in)
                        torch.nn.init.uniform_(m.bias.data, -bound, bound)

                elif initFunc == "Xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight.data)

                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(m.bias.data, -bound, bound)

                elif initFunc == "Xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight.data)

                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(m.bias.data, -bound, bound)

                elif initFunc == "Seed_Default":
                    torch.manual_seed(seed_num)
                    torch.nn.init.kaiming_uniform_(m.weight.data,a=sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_in)
                        torch.nn.init.uniform_(m.bias.data, -bound, bound)

                elif initFunc == "Uniform_end":
                    if (m.weight.data.size(0)) != 1:
                        m.weight.data.bernoulli_(0.5)
                        m.weight.data.mul_(2)
                        m.weight.data.sub_(1)
                        points_left = np.arange(-3, -2.5, 0.5 / (H / 2))
                        points_right = np.arange(2.5, 3, 0.5 / (H / 2))
                        for i in range(0, math.floor(m.bias.data.size(0) / 2)):
                            m.bias.data[i] = points_left[i]
                        for i in range(math.ceil(m.bias.data.size(0) / 2), m.bias.data.size(0)):
                            m.bias.data[i] = points_right[i - math.ceil(m.bias.data.size(0) / 2)]
                        m.weight.data = torch.nn.Parameter(m.weight.data)
                        m.bias.data = torch.nn.Parameter(m.bias.data)
                    else:
                        m.weight.data.bernoulli_(0.5)
                        m.weight.data.mul_(2)
                        m.weight.data.sub_(1)
                        m.bias.data.fill_(0.0)


        x_test = torch.FloatTensor(N, D_in).to(device)
        torch.linspace(MinInput, MaxInput, steps=N, out=x_test)


        # Use the nn package to define our model and loss function.
        if activationFunc == "ReLu":
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(H),
                torch.nn.Linear(H, D_out),
            )
        elif activationFunc == 'DeepReLu':
            if H_2 == 0 or numH == 0:
                model = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(H),
                    torch.nn.Linear(H, D_out),
                )
            else:
                class DeepNet(torch.nn.Module):
                    def __init__(self, D_in, H, H_2, D_out,numH):
                        super(DeepNet, self).__init__()
                        self.inputL = torch.nn.Linear(D_in, H)
                        self.hiddenL = torch.nn.Linear(H, H_2)
                        # self.hiddenH = torch.nn.Linear(H_2,H_2)
                        self.hiddenH = torch.nn.ModuleList([torch.nn.Linear(H_2,H_2) for _ in range(numH-1)])
                        self.outputL = torch.nn.Linear(H_2, D_out)
                        self.numH = numH-1
                        self.bn1 = torch.nn.BatchNorm1d(H)
                        self.bn2 = torch.nn.BatchNorm1d(H_2)
                        if initFunc != "Default":
                            weights_init(self.inputL)
                    def forward(self, x, getOut=False):

                        y_1 = self.bn1(F.relu(self.inputL(x)))
                        h_preRelu = self.hiddenL(y_1)
                        h_postRelu = F.relu(h_preRelu)
                        h_s = [(h_preRelu,h_postRelu)]
                        for i in range(self.numH):
                            h_preRelu = self.hiddenH[i](h_postRelu)
                            h_postRelu = F.relu(h_preRelu)
                            h_s.append((h_preRelu,h_postRelu))
                        y = self.outputL(h_postRelu)
                        if getOut:
                            return y, h_s
                        else:
                            return y
                model = DeepNet(D_in, H, H_2, D_out,numH).to(device)


        elif activationFunc == "Sigmoid":
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.Sigmoid(),
                torch.nn.Linear(H, D_out),
            )

        elif activationFunc == "LeakyReLU":
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(H, D_out),
            )
        elif activationFunc == "TanH":
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.Tanh(),
                torch.nn.Linear(H, D_out),
            )

        loss_fn = torch.nn.MSELoss(size_average=False)




        if initFunc != "Default" and activationFunc != "DeepReLu":
             torch.manual_seed(seed_num) #this line may be useless
             model.apply(weights_init)

        model.to(device)


        if freeze_v:
            #list(model.parameters())[0].requires_grad = False
            #list(model.parameters())[1].requires_grad = False
            list(model.parameters())[2].requires_grad = False
            list(model.parameters())[3].requires_grad = False


        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        changeLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20000, gamma = 1.0)
        #Some variables for plotting residue vs time.
        residue_data = {}

        #MiniBatch:
        class PrepareData(torch.utils.data.Dataset):

            def __init__(self, X, Y):
                self.x = X;
                self.y = y;

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        ds = PrepareData(X=x, Y=y)
        trainloader= torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True)

        #Initialize masks to all ones
        mask = {}
        percents ={}
        for w in range(len(list(model.parameters()))):
            name, para = list(model.named_parameters())[w]
            mask[name] = np.ones(para.shape)
            percents[name] = percent
        #test




        Loss = []
        Time_scale = []

        # fig = figure()


        # Plot Training Set
        xx = x.cpu().numpy()
        yy = y.cpu().numpy()
        # ax = axes(projection='3d')

        if not plot_residue and not plot_histograms and not plot_loss:
            # pass
            plot(xx, yy, 'g.', label='Train')
            # ax.scatter3D(x, y[:, 1], y[:, 0], 'gray')

        if mini_batch_option:
            for t in range(NumIters + 1):
                changeLR.step()
                for i, data in enumerate(trainloader):
                    x,y = data
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    loss_sc = loss.item() ** 0.5
                    # Zero out gradients (otherwise they are accumulated)
                    optimizer.zero_grad()
                    # Backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # Update parameters according to chosen optimizer
                    optimizer.step()

                if t % PrintFreq == 0:
                    print(t, loss.item(), loss_sc)

                # Plot Current Fitted NN Predictions on Test Set
                if (t % PlotFreq == 0) or (t % PruneFreq ==  0):
                    model.eval()  # go into inference mode, needed for DropOut, any Norm, anything where tr and test are different
                    y_test = model(x_test)
                    model.train()
                    # line_color = f'C{t // PlotFreq}'
                    line_color = cmap(t / NumIters)

                    if not plot_residue:
                        plot(x_test.data.cpu().numpy(), y_test.data.cpu().numpy(), 'r-', label=f'Test({t})', linewidth=1.0,
                             color=line_color)

                    modifier = 2
                    if plot_residue:
                        for i in range(-3 * modifier, 4 * modifier):
                            model.eval()
                            i = i / modifier
                            y_predicted = model(torch.Tensor([i]))
                            # print(i)
                            # print(y_predicted.item())
                            y_residue = y_predicted - (torch.sin(2 * torch.Tensor([i])))
                            if i not in residue_data:
                                residue_data[i] = OrderedDict()
                                residue_data[i][t] = abs(y_residue.item())
                            else:
                                residue_data[i][t] = abs(y_residue.item())

                            model.train()

        else:
            for t in range(NumIters+1):

                changeLR.step()
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(x)

                # Compute loss
                loss = loss_fn(y_pred, y)
                loss_sc = loss.item() ** 0.5

                # Print loss
                if t % PrintFreq == 0:
                    print(t, loss.item(), loss_sc)

                # Plot Current Fitted NN Predictions on Test Set

                if t == 0:
                    ws = (list(model.parameters())[0].data.cpu().numpy())
                    bs = (list(model.parameters())[1].data.cpu().numpy())
                    x_brk = torch.from_numpy(np.divide(-1 * bs, np.transpose(ws)[0])).to(device)
                    init_x_brk = x_brk.view(-1, 1)
                    model.eval()
                    init_y_brk = model(init_x_brk)
                    model.train()

                if t % PlotFreq == 0:
                    imgs = []
                    leg = []
                    #For plotting training loss vs epoch
                    Loss.append(loss.item())
                    Time_scale.append(t)

                    if plot_histograms:
                        ws = list(model.parameters())[0].data.cpu().numpy()
                        bs = list(model.parameters())[1].data.cpu().numpy()
                        breakPointsLocations = np.divide(np.squeeze(bs), np.squeeze(ws))
                        breakPointsLocations = breakPointsLocations * -1.0
                        temp = list(breakPointsLocations)
                        breakPointsData.append(temp)

                    model.eval() # go into inference mode, needed for DropOut, any Norm, anything where tr and test are different
                    if activationFunc == "DeepReLu" and (H_2 != 0 and numH !=0):
                        y_test, deep_layers = model(x_test,getOut)
                    else:
                        y_test = model(x_test)
                    model.train()
                    # line_color = f'C{t // PlotFreq}'
                    line_color = cmap(t / NumIters)




                    if makeVid:
                        ylim(-2, 2)
                        xlim(-5, 5)
                        ws = (list(model.parameters())[0].data.cpu().numpy())
                        bs = (list(model.parameters())[1].data.cpu().numpy())
                        x_brk = torch.from_numpy(np.divide(-1 * bs, np.transpose(ws)[0])).to(device)
                        x_brk = x_brk.view(-1, 1)
                        model.eval()
                        y_brk = model(x_brk)
                        model.train()
                        brk_plot = scatter(x_brk.data.cpu().numpy(), y_brk.data.cpu().numpy(),
                                         linewidth=1.0, color=line_color, animated=True)
                        leg = ['Ground Truth','Breakpoint Locations (Layer 1)']
                        ##secondary Breakpoints

                        cmap2 = matplotlib.cm.get_cmap('Greys')
                        cmap3 = matplotlib.cm.get_cmap('Reds')
                        cmap4 = matplotlib.cm.get_cmap("hsv")
                        if activationFunc == "DeepReLu" and getOut and plot_deep_brk:

                            layerIter = 0
                            tot_bps = H
                            for layer in deep_layers: #layer[0] is preReLu activation layer[1] is postReLu activation
                                bps = 0
                                layerIter += 1
                                x_brk_h = []
                                y_brk_h = []
                                sign_change = False
                                for unit in range((layer[0].data.shape[-1])):
                                    for iter in range(len(x_test) - 1):
                                        if layer[1].data[iter, unit] == 0 and \
                                                (layer[1].data[iter + 1, unit] > 0 or layer[1].data[iter - 1, unit] > 0):
                                            bps += 1
                                            x_brk_h.append(x_test.data.cpu().numpy()[iter])
                                            y_brk_h.append(layer[1].data.cpu().numpy()[iter,unit])
                                brk_plot_h = scatter(x_brk_h, y_brk_h, marker = 'x',c = [cmap4((layerIter+2)/(numH + 5))],
                                                 animated=True)
                                imgs.append(brk_plot_h)
                                brk_proj_h = scatter(x_brk_h, np.zeros(len(x_brk_h)) - (1.7 - .1 * layerIter), marker='|',c = [cmap4((layerIter+2)/(numH + 5))],
                                                   animated=True)
                                imgs.append(brk_proj_h)
                                tot_bps += bps
                                if (t%PrintFreq ==0):
                                    print(bps)
                                # leg.append('Breakpoint Locations (Layer '+ str(layerIter) + ')')
                            brk_plot_h = scatter(x_brk_h, y_brk_h, marker = 'x',c = [cmap4((layerIter+2)/(numH + 5))],
                                               animated=True)
                            imgs.append(brk_plot_h)
                            brk_proj_h = scatter(x_brk_h, np.zeros(len(x_brk_h)) - (1.7 - .1 * layerIter), marker='|',c = [cmap4((layerIter+2)/(numH + 5))],
                                               animated=True)
                            imgs.append(brk_proj_h)

                            for layer in deep_layers:
                                for unit in range((layer[0].data.shape[-1])):
                                    nn_plot_1, = plot(x_test.data.cpu().numpy(), layer[0].data.cpu().numpy()[:, unit], 'r-',
                                                      color=(cmap2(1/4 + .1*layerIter)), animated=True,linewidth=.5)
                                    nn_plot_2, = plot(x_test.data.cpu().numpy(), layer[1].data.cpu().numpy()[:, unit], 'k-',
                                                      color=(cmap3(3/4)), animated=True,linewidth=1)
                                    imgs.extend([nn_plot_1, nn_plot_2])
                                    # leg.extend([f'Unit {unit+1} PreReLu', f'Unit {unit} PostReLu'])

                            tot_piece = text(3.2, 1.35, 'pieces = ' + str(tot_bps))
                            imgs.append(tot_piece)
                        nn_plot, = plot(x_test.data.cpu().numpy(), y_test.data.cpu().numpy(), 'r-',
                                        linewidth=2.5, color=line_color, animated=True)

                        imgs.extend([nn_plot])


                        # legend(leg,loc='upper left')
                        iterno = text(3.2, 1.5, 'iter = ' + str(t))

                        if projectBrk:
                            brk_proj_x = scatter(x_brk.data.cpu().numpy(), np.zeros(H) - 1.7, marker = '|', color=cmap4((2)/(numH + 5)),
                                               animated=True)
                            brk_proj_y = scatter(np.zeros(H) - 4, y_brk.data.cpu().numpy(), marker = '_', color=cmap4((2)/(numH + 5)),
                                               animated=True)
                            imgs.extend([brk_proj_x, brk_proj_y])
                            # layerIter = 0
                            # for layer in deep_layers:
                            #     brk_proj_h, = plot(x_brk_h, np.zeros(len(x_brk_h)) - (1.6-.1*layerIter), 'ms',markersize =4,
                            #                        animated=True)
                            #     layerIter+=1
                            #     imgs.append(brk_proj_h)

                        imgs.extend([brk_plot,iterno])
                        ims.append(tuple(imgs))


                    if not plot_residue and not plot_histograms and not plot_loss and not makeVid:
                         plot(x_test.data.cpu().numpy(), y_test.data.cpu().numpy(), 'r-', label=f'Test({t})', linewidth=1.0, color=line_color)

                    modifier = 2
                    if plot_residue:
                        for i in range(-3 * modifier,3 * modifier + 1):
                            model.eval()
                            i = i * 1.0 / modifier
                            y_predicted = model(torch.Tensor([i]).reshape(1,1))
                            y_residue = y_predicted - (torch.sin(2*torch.Tensor([i])))
                            if i not in residue_data:
                                residue_data[i] = OrderedDict()
                                residue_data[i][t] = abs(y_residue.item())
                            else:
                                residue_data[i][t] = abs(y_residue.item())

                            model.train()
                if iter_prune:
                    if (t % PruneFreq ==0) & (t>1) & (t<NumIters/2):
                        mask = prune_by_percent(percents, mask, dict(model.named_parameters()))
                        torch.manual_seed(seed_num)
                        model.apply(weights_init)
                        for layer, val in mask.items():
                            print(layer, 'Nonzero Units =', np.sum(val), 'Units =', len(val))
                            dict(model.named_parameters())[layer].data *= torch.Tensor(val).to(device)
                if grow_prune:
                    if (t % PruneFreq == 0) & (t > 1) & ((t < NumIters / 2) or (t>NumIters*3/4)):
                        if activationFunc == "DeepReLu" and getOut:
                           lastlayer = deep_layers[-1]
                           valueadded = torch.sum(lastlayer[1].data,0)
                           low_unit = torch.argsort(valueadded)[0]
                           error = np.square((y_pred - y).detach().cpu().numpy())
                           max_errors = np.argsort(np.transpose(error))[0][-1:]
                           max_x = x[max_errors].cpu().numpy()
                           y_value = lastlayer[0].data[max_errors,low_unit]
                           laststr = numH-2
                           dict(model.named_parameters())[f'hiddenH.{laststr}.bias'].data += -1*y_value

                        else:
                            new_mask = prune_by_percent(percents, mask, dict(model.named_parameters()))
                            for layer,val in new_mask.items():
                                if "0.weight" in layer:
                                    dict(model.named_parameters())[layer].data *= torch.Tensor(val).to(device)

                            error = np.square((y_pred- y).detach().cpu().numpy())
                            prunedNum = len(new_mask['0.weight']) - np.sum(new_mask['0.weight'])
                            max_errors = np.argsort(np.transpose(error))[0][-1*int(prunedNum):]
                            max_x = x[max_errors].cpu().numpy()
                            gen_grad = pwgg(new_mask,max_x,dict(model.named_parameters()))

                            for layer, val in gen_grad.items():
                                if "0.weight" in layer:
                                    dict(model.named_parameters())[layer].data += torch.Tensor(val).to(device)


                # Zero out gradients (otherwise they are accumulated)
                optimizer.zero_grad()
                        # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                if iter_prune:
                    if t>1:
                        for layer, val in mask.items():
                            dict(model.named_parameters())[layer].grad *= torch.Tensor(val).to(device)



                # Update parameters according to chosen optimizer
                optimizer.step()




        if plot_histograms:
            global d
            number_of_frames = len(breakPointsData)

            def update_hist(num, data):
                plt.cla()
                kde = stats.gaussian_kde(data[num])
                plt.hist(data[num],density=True, bins=12, range=(-3, 3),edgecolor='black', linewidth=1.2)
                xx = np.arange(-3,3,0.1)
                kde.set_bandwidth(50.0/H)
                plt.plot(xx,kde(xx),linewidth = 1.0)
                d.set_text(f'"Epoch = {num * 1000}"')
                plt.ylabel('Distribution')
                plt.title(
                    f'{activationFunc} Net ({D_in},{H},{D_out}) trained for {NumIters} Iters w/ Learning Rate={learning_rate} w/ Initialization = {initFunc}')
                plt.draw()

            fig = plt.figure()
            hist = plt.hist(breakPointsData[0], density=True, bins=12, range=(-3, 3),edgecolor='black', linewidth=1.2)
            d = plt.figtext(0.8, 0.8, "")
            plt.ylabel('Distribution')
            title(
                f'{activationFunc} Net ({D_in},{H},{D_out}) trained for {NumIters} Iters w/ Learning Rate={learning_rate} w/ Initialization = {initFunc}')

            ani = animation.FuncAnimation(fig, update_hist, frames = number_of_frames, fargs=(breakPointsData,),interval = 500, blit=False, repeat=False, save_count=number_of_frames)

        #Plot residue data
        if (plot_residue):
            N = 6.0
            for key in residue_data:
                lists = sorted(residue_data[key].items())  # sorted by key, return a list of tuples
                x, y = zip(*lists)  # unpack a list of pairs into two tuples
                x = list(x)
                y = list(y)
                v = float(float(key + 3.0)/ N)
                c = cmap(v)
                semilogy(x,y,'-r',label=f'x = {key}',linewidth = 1.0,color = c)

        # Plot NN Predictions on Training Set

        #yy_p = y_pred.detach().numpy()
        #plot(xx, yy_p, 'b.', label='Pred')
        #plot(x_test.detach().numpy(), y_test.detach().numpy(), 'r-', label='Test')

        if plot_loss:
            # semilogy(Time_scale, Loss, '-r',linewidth = 1.0,label = f"H_2 = {H_2}",c=cmap4((count+1)/8))
            # pass
            iters.append(iter)
            losses.append(loss.item())

        print(H, H_2, 'done')
        # Finalize Plots
        xlabel('Input')
        ylabel('Output')

semilogy(iters,losses,'.')
#Brk diff

# ws = (list(model.parameters())[0].data.cpu().numpy())
# bs = (list(model.parameters())[1].data.cpu().numpy())
# x_brk = torch.from_numpy(np.divide(-1 * bs, np.transpose(ws)[0])).to(device)
# x_brk = x_brk.view(-1, 1)
# model.eval()
# y_brk = model(x_brk)
# brk_diff_x = torch.abs(init_x_brk - x_brk)
# brk_diff_y = torch.abs(init_y_brk - y_brk)
# clf()
# plot(init_x_brk.cpu().numpy(),brk_diff_x.cpu().numpy(),'ro')
#
# xlabel('Initial Break Points')
# ylabel('Change in X')
# ylim(-2, 2)
# xlim(-5, 5)

if 'Deep' in activationFunc:
    netString = ''
    for _ in range(numH):
        netString += ','+ str(H_2)

    title(f'{activationFunc} Net ({D_in},{H}{netString},{D_out}) trained for {NumIters} Iters w/ LR={learning_rate} w/ Init = {initFunc}')
    title(f'{activationFunc} Net (Params = {H_tot}) trained for {NumIters} Iters w/ Learning Rate={learning_rate} w/ Initialization = {initFunc}')
else:
    title(f'{activationFunc} Net ({D_in},{H},{D_out}) trained for {NumIters} Iters w/ Learning Rate={learning_rate} w/ Initialization = {initFunc}')


legend()
if plot_curve:
    fname = f'./Figures/{tag}_fixedv_fitted_nn_snapshots_H={H}_T={NumIters}_lr={learning_rate}_N={N}_Activation={activationFunc}_Initialization = {initFunc}_minibatch = {mini_batch_option}_1.pdf'

elif plot_residue:
    fname = f'./Figures/semigLog_fixedV_Residue_vs_time_fitted_nn_snapshots_H={H}_T={NumIters}_lr={learning_rate}_N={N}_Activation={activationFunc}_Initialization = {initFunc}_minibatch = {mini_batch_option}.pdf'

elif plot_histograms:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fname = f'./Figures/KDE_Histograms_bp_distribution_BN(AFTER_RELU)_H={H}_T={NumIters}_lr={learning_rate}_N={N}_Activation={activationFunc}_Initialization = {initFunc}_minibatch = {mini_batch_option}_bin_width = 0.1_2'
    ani.save(fname + '.mp4', writer = writer)
    show()

elif plot_loss:
    fname = f'./Figures/semigLog_BN(before RELU)_Training_loss_vs_time_fitted_nn_snapshots_H={H}_T={NumIters}_lr={learning_rate}_N={N}_Activation={activationFunc}_Initialization = {initFunc}_minibatch = {mini_batch_option}.pdf'
    xlabel('Num in H')
    ylabel('Loss')
    # savefig(fname)

elif makeVid:
    mywriter = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=18000)
    # fname = f'./Figures/Animated_curve_fitted_nn_snapshots_H={H}_T={NumIters}_lr={learning_rate}_N={N}_Activation={activationFunc}_Initialization = {initFunc}_minibatch = {mini_batch_option}_'+tag + f'percent_prune= {percent}'
    fname = './Figures/'+tag #change later
    ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
    ani.save(fname+'.mp4', writer=mywriter)
    # print(fname)

fname = './Figures/'+tag
savefig(fname)
print(fname)
import sys
sys.exit()
# show()