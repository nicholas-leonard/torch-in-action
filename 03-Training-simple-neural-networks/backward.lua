require "nn"

bce = nn.BCECriterion()

input = torch.Tensor({0.6})
target = torch.Tensor({0})

loss = bce:updateOutput(input, target)
gradInput = bce:updateGradInput(input, target)

print("before: loss="..loss.."; gradInput="..gradInput[1])

input:add(-0.1, gradInput)

loss = bce:updateOutput(input, target)
gradInput = bce:updateGradInput(input, target)

print("after: loss="..loss.."; gradInput="..gradInput[1])

--[[
before: loss=0.91629073187166; gradInput=2.4999999999833	
after: loss=0.43078291609348; gradInput=1.5384615384564	
--]]
