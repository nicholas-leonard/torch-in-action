require "nn"
require "optim"

input = torch.Tensor({{0,0},{0,1},{1,0},{1,1}})
target = torch.Tensor({{0},{1},{1},{0}})

hiddensize = 2
mlp = nn.Sequential()
 :add(nn.Linear(2, hiddensize))
 :add(nn.Tanh())
 :add(nn.Linear(hiddensize, 1))
 :add(nn.Sigmoid())

bce = nn.BCECriterion()

for i=1,1000 do
   -- sample
   local idx = math.random(1,4)
   local x, y = input[idx], target[idx]

   -- forward
   local y_hat = mlp:forward(x)
   local loss = bce:forward(y_hat, y)

   -- backward
   local grad_y_hat = bce:backward(y_hat, y)
   mlp:zeroGradParameters()
   mlp:backward(x, grad_y_hat)

   -- update
   mlp:updateParameters(0.1)
end

-- confusion matrix
cm = optim.ConfusionMatrix(2)

for i=1,4 do
   local x, y = input[i], target[i]
   local y_hat = mlp:forward(x)
   cm:add(y_hat[1] > 0.5 and 2 or 1, y[1]+1)
end

--print(cm)
cm:updateValids()
assert(cm.totalValid == 1, "Run the script again until you get 100% accuracy")

-- get the classification boundary curve

print("x1, x2, y_hat")
input = torch.Tensor(2)
for x1=0,1,0.01 do
   for x2=0,1,0.01 do
      input[1], input[2] = x1, x2
      local output = mlp:forward(input)[1]
      if output < 0.52 and output > 0.48 then
         print(string.format("%f, %f, %f", x1, x2, output))
      end
   end
end
