require 'paths'
require 'torch'
require 'image'

local datapath = 'mnist'

local testimage = 't10k-images-idx3-ubyte'
local testlabel = 't10k-labels-idx1-ubyte'
local trainimage = 'train-images-idx3-ubyte'
local trainlabel = 'train-labels-idx1-ubyte'

local labelpath = paths.concat(datapath, testlabel)
assert(paths.filep(labelpath))

local file = io.open(labelpath, "r")

local data = file:read("*a")
print(#data)

local labels = data:sub(-10000,-1)
print(#labels)

local targets = torch.LongTensor(#labels):fill(-1)
for i=1,#labels do
   targets[i] = labels:byte(i)
end

assert(targets:min() ~= -1)

targets:add(1) -- 0-9 -> 1,10

file:close()

local imagepath = paths.concat(datapath, testimage)
local file = io.open(imagepath)

local data = file:read("*a")
print(#data)

local images = data:sub(16+1, -1)
print(#images, #images/(28*28))

local inputs = torch.ByteTensor(#labels, 1, 28, 28)

local ffi = require 'ffi'
local idata = inputs:data()
ffi.copy(idata, images)

inputs = inputs:float()

local indices = torch.LongTensor(16):random(1,#labels)
local samples = inputs:index(1, indices)

local display = image.toDisplayTensor(samples, 2, 4)
print(display:size())
image.save("samples/mnistsamples.png", display)

