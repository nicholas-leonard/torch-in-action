require "nn"
local dl = require "dataload"
require "optim"

trainset = dl.loadMNIST()

model = nn.Sequential()
   :add(nn.View(28*28))
   :add(nn.Linear(28*28, 10))
   :add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

model:float(); criterion:float()

print("Epoch; Average Loss")

-- optimize model
for epoch=1,200 do
   local sumloss, count = 0, 0
   for i, input, target in trainset:sampleiter(32, 10000) do
      local output = model:forward(input)
      sumloss = sumloss + criterion:forward(output, target)
      count = i

      local gradOutput = criterion:backward(output, target)
      model:zeroGradParameters()
      model:backward(input, gradOutput)

      model:updateParameters(0.1)
   end
   local avgloss = sumloss/count
   print(string.format("%d; %f", epoch, avgloss))
   if avgloss < 0.007 then
      break
   end
end

-- evaluate empirical risk and confusion matrix
cm = optim.ConfusionMatrix(10)
sumloss, count = 0, 0
for i, input, target in trainset:subiter(32) do
   local output = model:forward(input)
   sumloss = sumloss + criterion:forward(output, target)
   cm:batchAdd(output, target)
   count = i
end
assert(count == 50000)
print(cm)
print("Avg NLL:"..sumloss/count)

model:clearState()
torch.save("logreg-mnist.t7", model)
