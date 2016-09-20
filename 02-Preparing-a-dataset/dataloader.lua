require "paths"
require "torch"
local dl = {}                

-- LISTING 2.1: abstract DataLoader class 
                               
local DataLoader = torch.class('dl.DataLoader', dl)                 

function DataLoader:index(indices, inputs, targets, ...)          
   error"Not Implemented"
end

function DataLoader:sample(batchsize, inputs, targets, ...)     
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchsize):random(1,self:size())
   return self:index(self._indices, inputs, targets, ...)
end

function DataLoader:sub(start, stop, inputs, targets, ...)   
   self._indices = self._indices or torch.LongTensor()
   self._indices:range(start, stop)
   return self:index(self._indices, inputs, targets, ...)
end

function DataLoader:size()
   error"Not Implemented"
end

-- LISTING 2.5: random sample iterator for the DataLoader class

function DataLoader:sampleiter(batchsize, epochsize, ...)
   batchsize = batchsize or 32                                       
   epochsize = epochsize or -1 
   epochsize = epochsize > 0 and epochsize or self:size()
   local dots = {...}                                                

   local nsampled = 0
   local inputs, targets

   return function()                                                 
      if nsampled >= epochsize then
         return
      end
      
      local bs = math.min(nsampled+batchsize, epochsize) - nsampled   
      
      inputs, targets = self:sample(bs, inputs, targets, unpack(dots))
      
      nsampled = nsampled + bs
      return nsampled, unpack(batch)    
   end
end	

-- LISTING 2.2: concrete TensorLoader class

local TensorLoader = torch.class('dl.TensorLoader','dl.DataLoader',dl)

function TensorLoader:__init(inputs, targets)                        
   self.inputs = inputs
   self.targets = targets
   local message = "number of input and target samples must match"
   assert(self.inputs:size(1) == self.targets:size(1), message)
end

function TensorLoader:index(indices, inputs, targets)               
   inputs = inputs or self.inputs.new()
   targets = targets or self.targets.new()
   inputs:index(self.inputs, 1, indices)
   targets:index(self.targets, 1, indices)
   return inputs, targets
end

function TensorLoader:size()                                       
   return self.inputs:size(1)
end 

-- LISTING 2.3: define the MNIST loader function

function dl.loadMNIST(datapath)
   local train = {                                               
      inputs = torch.load(paths.concat(datapath, "traininputs.t7")),
      targets = torch.load(paths.concat(datapath, "traintargets.t7"))
   }
   local test = {
      inputs = torch.load(paths.concat(datapath, "testinputs.t7")),
      targets = torch.load(paths.concat(datapath, "testtargets.t7"))
   }

   local valid = {}                                                  
   valid.inputs = train.inputs:sub(50001,60000)
   valid.targets = train.inputs:sub(50001,60000)

   train.inputs = train.inputs:sub(1,50000)
   train.targets = train.targets:sub(1,50000)

   train = dl.TensorLoader(train.inputs, train.targets)             
   valid = dl.TensorLoader(valid.inputs, valid.targets)
   test = dl.TensorLoader(test.inputs, test.targets)
   
   return train, valid, test
end

return dl                                         
