require 'paths'

datapath = 'mnist'
inputs = torch.load(paths.concat(datapath, 'testinputs.t7'))
targets = torch.load(paths.concat(datapath, 'testtargets.t7'))

dl = dofile "dataloader.lua"
testset = dl.TensorLoader(inputs, targets)
assert(testset.inputs and testset.targets)

src = torch.Tensor(1,5):range(2,6)
print(src)

indices = torch.LongTensor({5,2,3})
res = src.new()
res:index(src, 2, indices)
print(res)

inputs, targets = testset:index(torch.LongTensor({1,2,3}))
print(inputs:size())
print(targets:size())

print(testset:size())

inputs, targets = testset:sub(1, 3)
inputs, targets = testset:sample(10)
