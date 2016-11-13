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
local opt = cmd:parse(arg or {})

-- process cmd-line options
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.epochsize = opt.epochsize > 0 and opt.epochsize or nil

-- load training set
local trainset, validset = dl.loadMNIST()

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

-- confusion matrix used for training and cross-valiation
local validcm = optim.ConfusionMatrix(10)
local traincm = optim.ConfusionMatrix(10)

-- optimize model using SGD
print("Epoch, Train error, Valid error")
for epoch=1,opt.maxepoch do

   -- 1. training
   traincm:zero()
   model:training()
   for i, input, target in trainset:sampleiter(opt.batchsize, opt.epochsize) do
      local output = model:forward(input)
      traincm:batchAdd(output, target)

      criterion:forward(output, target)
      local gradOutput = criterion:backward(output, target)
      model:zeroGradParameters()
      model:backward(input, gradOutput)
      model:updateParameters(opt.lr)
   end
   traincm:updateValids()
   opt.trainerr = 1 - traincm.totalValid

   -- 2. cross-validation
   validcm:zero()
   model:evaluate()
   for i, input, target in validset:subiter(opt.batchsize) do
      local output = model:forward(input)
      validcm:batchAdd(output, target)
   end
   validcm:updateValids()
   opt.validerr = 1 - validcm.totalValid

   print(string.format("%d, %f, %f", epoch, opt.trainerr, opt.validerr))

end

model:clearState()
model.opt = opt
torch.save("mlp-mnist-crossvalidate.t7", model)

