--[[
We will use the MNIST data loader created in chapter 2. 
Listing 4.1 will extend listing 3.2 to use multiple layers. 
Everything else will be the same. 
It will be explained how the model capacity is increased to allow for 
non-linear discrimination and demonstrate this via better performance. 
We will include a diagram of the an MLP showing the usual connectionist view of a NN. 
Listing 4.1 will also move out the hyper-parameters to the command-line and introduce these. 
We will have a diagram plotting the training set learning curves of the logistic regression model side by side with the MLP. 

Changes:
-- everything is local
-- cmd-line args
-- epochsize
-- transfer
-- clear-state
-- nn.Convert
--]]

require "nn"
local dl = require "dataload"
require "optim"
require "dpnn" -- needed for nn.Convert

-- options : hyper-parameters and such
local cmd = torch.CmdLine() 
cmd:text()
cmd:text('Training a multi-layer perceptron on MNIST')
cmd:text('Options:')
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-batchsize', 32, 'number of samples per batch')
cmd:option('-epochsize', -1, 'number of samples per epoch')
cmd:option('-hiddensize', '{200,200}', 'number of hidden units')
cmd:option('-transfer', 'ReLU', 'non-linear transfer function')
cmd:option('-maxepoch', 200, 'stop after this many epochs')
cmd:option('-minloss', 0.0001, 'stop when training set loss is lower than this')
local opt = cmd:parse(arg or {})

-- process cmd-line options
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.epochsize = opt.epochsize > 0 and opt.epochsize or nil

-- load training set
local trainset = dl.loadMNIST()

-- define model and criterion
local inputsize = 28*28

local model = nn.Sequential()
model:add(nn.Convert())
model:add(nn.View(inputsize))

for i,hiddensize in ipairs(opt.hiddensize) do
   model:add(nn.Linear(inputsize, hiddensize))
   model:add(nn[opt.transfer]())
   inputsize = hiddensize
end

model:add(nn.Linear(inputsize, 10))
model:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

-- optimize model using SGD
print("Epoch; Average Loss")
for epoch=1,opt.maxepoch do
   local sumloss, count = 0, 0
   for i, input, target in trainset:sampleiter(opt.batchsize, opt.epochsize) do
      local output = model:forward(input)
      sumloss = sumloss + criterion:forward(output, target)
      count = i
      
      local gradOutput = criterion:backward(output, target)
      model:zeroGradParameters()
      model:backward(input, gradOutput)

      model:updateParameters(opt.lr)
   end
   local avgloss = sumloss/count
   print(string.format("%d; %f", epoch, avgloss))
   if avgloss < opt.minloss then
      break
   end
end

-- evaluate empirical risk and confusion matrix
local cm = optim.ConfusionMatrix(10)
local sumloss, count = 0, 0
for i, input, target in trainset:subiter(opt.batchsize) do
   local output = model:forward(input)
   sumloss = sumloss + criterion:forward(output, target)
   cm:batchAdd(output, target)
   count = i
end
print(cm)
print("Avg NLL:"..sumloss/count)

model:clearState()
torch.save("mlp-mnist.t7", model)
