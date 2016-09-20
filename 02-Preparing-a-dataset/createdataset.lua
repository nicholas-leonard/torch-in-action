require 'paths'
require 'torch'
require 'image'

local datapath = 'mnist'

local test = {image='t10k-images-idx3-ubyte', label='t10k-labels-idx1-ubyte', size=10000}
local train = {image='train-images-idx3-ubyte', label='train-labels-idx1-ubyte', size=60000}

for prefix, dataset in pairs{test=test, train=train} do
   local labelpath = paths.concat(datapath, dataset.label)
   assert(paths.filep(labelpath))

   local file = io.open(labelpath, "r")

   local data = file:read("*a")
   print(#data)

   local labels = data:sub(-dataset.size,-1)
   print(#labels)

   local targets = torch.LongTensor(#labels):fill(-1)
   for i=1,#labels do
      local class = labels:byte(i)
      print(type(class), class)
      targets[i] = class
   end

   assert(targets:min() ~= -1)

   targets:add(1) -- 0-9 -> 1,10

   file:close()

   local imagepath = paths.concat(datapath, dataset.image)
   local file = io.open(imagepath)

   local data = file:read("*a")
   print(#data)

   local images = data:sub(16+1, -1)
   print(#images, #images/(28*28))

   a = torch.Timer()
   local inputs = torch.ByteTensor(#labels, 1, 28, 28)

   --[[
   for i=1,#labels do
      for j=1,28 do
         for k=1,28 do 
            local idx = (i-1)*28*28 + (j-1)*28 + k
            print(idx)
            inputs[{i,j,k}] = images:byte(idx)
         end
      end
   end--]]

   local ffi = require 'ffi'
   local idata = inputs:data()
   ffi.copy(idata, images)

   inputs = inputs:float()
   print("ffi for", a:time().real)

   image.save(paths.concat(datapath, prefix..'.jpg'), inputs[1])


   a = torch.Timer()
   local inputs = torch.FloatTensor(#labels, 1, 28, 28)
   local storage = inputs:storage()

   for idx=1,#images do
      storage[idx] = images:byte(idx)
   end
   print("single for", a:time().real)

   -- save to disk
   torch.save(paths.concat(datapath, prefix..'inputs.t7'), inputs)
   torch.save(paths.concat(datapath, prefix..'targets.t7'), targets)

   --[[
   local inputs = torch.load(paths.concat(datapath, prefix..'inputs.t7'))
   local targets = torch.load(paths.concat(datapath, prefix..'targets.t7'))
   --]]

   math.randomseed(89898)

   for i=1,3 do
      local sampleidx = math.random(1,inputs:size(1))
      local input = inputs[sampleidx]
      local target = targets[sampleidx]
      local filename = string.format("samples/sample%d_class%d.png", sampleidx, target)
      image.save(filename, input) 
   end
end

