-- Note for Mac OS X users:
-- if you get a 'libjpeg' error try :
-- $ brew install libjpec
-- $ luarocks make image

-- LISTING 1.1: Load image classification dataset into tensors

require "paths"
require "image"
local N, depth, height, width = 4, 3, 28, 28
local datapath = "facedetect/"
local inputs = torch.DoubleTensor(N, depth, height, width):zero()
local targets = torch.LongTensor(N):zero()
local classes = {"face","background"}
local n = 0
for classid=1,2 do
   local class = classes[classid]
   local classpath = paths.concat(datapath, class)
   for imagefile in paths.iterfiles(classpath) do
      n = n + 1
      local imagetensor = image.load(paths.concat(classpath, imagefile))
      image.scale(inputs[n], imagetensor)
      targets[n] = classid
   end
end
assert(n == N, "Missing samples")

-- LISTING 1.2: Assembling a model and loss function for image classification

require 'nn'
require 'dpnn'

-- model is a convolutional neural network :
model = nn.Sequential()
-- 2 conv layers:
model:add(nn.Convert())
model:add(nn.SpatialConvolution(3, 16, 5, 5, 1, 1, 2, 2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.SpatialConvolution(16, 32, 5, 5, 1, 1, 2, 2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- 1 dense hidden layer:
outsize = model:outside{1,depth,height,width}
model:add(nn.Collapse(3))
model:add(nn.Linear(outsize[2]*outsize[3]*outsize[4], 200))
model:add(nn.ReLU())
-- output layer:
model:add(nn.Linear(200, 10))
model:add(nn.LogSoftMax())

-- loss function is negative log likelihood:
criterion = nn.ClassNLLCriterion()

-- LISTING 1.3: Neural network training using Stochastic Gradient Descent for 100 epoch

for epoch=1,100 do
   local sumloss = 0
   local N = inputs:size(1)
   for i=1,N do
      -- 1. sample one input and target pair from dataset
      local idx = torch.random(1,N)
      local input = inputs[idx]
      local target = targets:narrow(1,idx,1)
      -- 2. forward
      local output = model:forward(input)
      local loss = criterion:forward(output, target)
      sumloss = sumloss + loss
      -- 3. backward
      local gradOutput = criterion:backward(output, target)
      model:zeroGradParameters()
      local gradInput = model:backward(input, gradOutput)
      -- 4. Update
     model:updateParameters(0.1)
   end
   print("Epoch #"..epoch..": mean training loss = "..sumloss/N)
end
torch.save("facedetector.t7", model)
