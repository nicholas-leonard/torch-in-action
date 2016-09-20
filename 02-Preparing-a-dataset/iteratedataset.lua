require 'nn'
require 'dpnn'

-- returns a model an criterion for the MNIST dataset
function getModelCriterionMNIST()
   -- model and criterion implement multinomial logistic regression
   local model = nn.Sequential()
      :add(nn.View(28*28))
      :add(nn.Linear(28*28, 10))
      :add(nn.LogSoftMax())
   
   local criterion = nn.ClassNLLCriterion()
   
   -- cast to float to work with dataset
   model:float(); criterion:float()
   
   return model, criterion
end

datapath = "mnist"

-- LISTING 2.4: Using a TensorLoader to iterate through a training set

local dl = dofile "dataloader.lua"                                   
local trainset = dl.loadMNIST(datapath)                

local batchsize, epochsize = 32, trainset:size()                    

function ftrain(model, criterion, inputs, targets)                  
   local outputs = model:forward(inputs)
   local loss = criterion:forward(outputs, targets)
   local gradOutputs = criterion:backward(outputs, targets)
   model:zeroGradParameters()
   model:backward(inputs, gradOutputs)
   model:updateParameters(0.1)
end

local model, criterion = getModelCriterionMNIST()                 

local inputs, targets                                               
for i=1,epochsize/batchsize do                                   
   inputs, targets = trainset:sample(batchsize, inputs, targets)
   ftrain(model, criterion, inputs, targets)
   print("training batch: ", i)
end              
