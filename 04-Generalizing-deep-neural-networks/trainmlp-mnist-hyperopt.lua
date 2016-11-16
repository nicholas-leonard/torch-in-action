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
cmd:option('-earlystop', 20, 'max #epochs to find a better minima for early-stopping')
cmd:option('-weightdecay', 1e-5, 'weight decay regularization factor')
cmd:option('-savepath', paths.concat(dl.SAVE_PATH, 'mnist'), 'path to directory where to save model and learning curves')
cmd:option('-id', dl.uniqueid(), 'id string of this experiment (defaults to a unique id)')
cmd:option('-progress', false, 'print progress bar')
local opt = cmd:parse(arg or {})

-- load training set
local trainset, validset = dl.loadMNIST()

-- process cmd-line options
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.epochsize = opt.epochsize > 0 and opt.epochsize or trainset:size()
opt.version = 1
opt.version = 2 -- uses dpnn's Module:weightdecay()

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

-- confusion matrix used for cross-valiation
local validcm = optim.ConfusionMatrix(10)
local traincm = optim.ConfusionMatrix(10)
local ntrial, minvaliderr = 0, 1

-- comma separated value
paths.mkdir(opt.savepath)
local csvpath = paths.concat(opt.savepath, opt.id..'.csv')

local csvfile = io.open(csvpath, 'w')
csvfile:write("Epoch,train error,valid error\n")

-- optimize model using SGD
for epoch=1,opt.maxepoch do
   print("\n"..opt.id.."; epoch #"..epoch.." :")

   -- 1. training
   local timer = torch.Timer()
   traincm:zero()
   for i, input, target in trainset:sampleiter(opt.batchsize, opt.epochsize) do
      local output = model:forward(input)
      criterion:forward(output, target)

      traincm:batchAdd(output, target)

      local gradOutput = criterion:backward(output, target)
      model:zeroGradParameters()
      model:backward(input, gradOutput)

      model:weightDecay(opt.weightdecay) -- weight decay

      if opt.progress then
         xlua.progress(math.min(i, opt.epochsize), opt.epochsize)
      end

      model:updateParameters(opt.lr)
   end
   traincm:updateValids()
   opt.trainerr = 1 - traincm.totalValid

   local speed = opt.epochsize/timer:time().real
   print(string.format("Speed : %f samples/second ", speed))
   print(string.format("Training error: %f", opt.trainerr))

   -- 2. cross-validation
   validcm:zero()
   for i, input, target in validset:subiter(opt.batchsize) do
      local output = model:forward(input)
      validcm:batchAdd(output, target)
   end
   validcm:updateValids()
   opt.validerr = 1 - validcm.totalValid

   print(string.format("Validation error: %f", opt.validerr))
   csvfile:write(string.format('%d,%f,%f\n', epoch, opt.trainerr, opt.validerr))

   -- 3. early-stopping
   ntrial = ntrial + 1
   if opt.validerr < minvaliderr then
      print("Found new minimum after "..ntrial.." epochs")
      minvaliderr = opt.validerr
      model.opt = opt
      model:clearState()
      local filename = paths.concat(opt.savepath, opt.id..'.t7')
      torch.save(filename, model)
      ntrial = 0
   elseif ntrial >= opt.earlystop then
      print("No new minima found after "..(epoch-ntrial).." epochs.")
      print("Lowest validation error: "..(minvaliderr*100).."%")
      print("Stopping experiment.")
      break
   end

end

csvfile:close()
print("CSV file saved to "..csvpath)
