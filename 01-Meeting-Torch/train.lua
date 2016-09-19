-- hyper-parameters

N = 4
depth = 3
height = 28
width = 28
savepath = "model.t7"

-- LISTING 1.1: Load image classification dataset into tensors

require "paths"
require "image"
local datapath = "facedetect/"
local inputs = torch.DoubleTensor(N, depth, height, width):zero()
local targets = torch.LongTensor(N):zero()
local classes = {"face","background"}
local n, classid = 0, 1
for classid=1,2 do
   local class = classes[classid]
   local classpath = paths.concat(datapath, class)
   for imagefile in paths.iterfiles(classpath) do
      n = n + 1
      local imagetensor = image.load(paths.concat(classpath, imagefile))
      image.scale(inputs[n], imagetensor)
      targets[n] = classid
   end
   classid = classid + 1
end
assert(n == N, "Missing samples")

-- LISTING 1.2: Assembling a model and loss function for image classification

require 'nn'
require 'dpnn'

-- model is a convolutional neural network :
module = nn.Sequential()
-- 2 conv layers:
module:add(nn.Convert())
module:add(nn.SpatialConvolution(3, 16, 5, 5, 1, 1, 2, 2))
module:add(nn.ReLU())
module:add(nn.SpatialMaxPooling(2, 2, 2, 2))
module:add(nn.SpatialConvolution(16, 32, 5, 5, 1, 1, 2, 2))
module:add(nn.ReLU())
module:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- 1 dense hidden layer:
outsize = module:outside{1,depth,height,width} 
module:add(nn.Collapse(3))
module:add(nn.Linear(outsize[2]*outsize[3]*outsize[4], 200))
module:add(nn.ReLU())
-- output layer:
module:add(nn.Linear(200, 10))
module:add(nn.LogSoftMax())

-- loss function is negative log likelihood:
criterion = nn.ClassNLLCriterion()

-- LISTING 1.3: Neural network training using Stochastic Gradient Descent for 100 epoch

for epoch=1,100 do
   local sumloss = 0
   local N = inputs:size(1)
   for i=1,N do
      -- 0. sample one input and target pair from dataset
      local idx = torch.random(1,N)
      local input = inputs[idx]
      local target = targets:narrow(1,idx,1)
      -- 1. forward
      local output = module:forward(input)
      local loss = criterion:forward(output, target)
      sumloss = sumloss + loss
      -- 2. backward
      local gradOutput = criterion:backward(output, target)
      module:zeroGradParameters()
      local gradInput = module:backward(input, gradOutput)
      -- 3. Update
     module:updateParameters(0.1) 
   end
   print("Epoch #"..epoch..": mean training loss = "..sumloss/N)
end
torch.save(savepath, module)
