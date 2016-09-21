require "nn"

input = torch.Tensor({{0,0},{0,1},{1,0},{1,1}})
target = torch.Tensor({{0},{1},{1},{1}})

logreg = nn.Sequential()
logreg:add(nn.Linear(2, 1))
logreg:add(nn.Sigmoid())

bce = nn.BCECriterion()

for i=1,1000 do
   -- sample
   local idx = math.random(1,4)
   local x, y = input[idx], target[idx]

   -- forward
   local y_hat = logreg:forward(x)
   local loss = bce:forward(y_hat, y)
   print(string.format("x={%d, %d}, y=%d, f(x)=%4.2f, L=%4.2f", x[1], x[2], y[1], y_hat[1], loss))

   -- backward
   local grad_y_hat = bce:backward(y_hat, y)
   logreg:zeroGradParameters()
   logreg:backward(x, grad_y_hat)

   -- update
   logreg:updateParameters(0.1)
end
