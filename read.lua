require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'

require 'util.Squeeze'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text('Options:')
-- task
cmd:option('-model','modelNotFound','path to the model')
cmd:text()
-- parse input params
opt = cmd:parse(arg)
print(opt.model)

-- load model
local checkpoint = torch.load(opt.model)
opt = checkpoint.opt
local protos = checkpoint.protos
protos.rnn:evaluate()
local idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)

-- the initial state of the cell/hidden states
local init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(2, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

function split2words(text)
	local words = {}
	for word in text:gmatch'([^%s]+)' do
		 table.insert(words,word)
	end
	return words
end

function word2x(word)
	local y = word2idx[word]
	if y == nil then y = word2idx[opt.tokens.UNK] end
	local x = torch.Tensor(2,24):fill(char2idx[opt.tokens.ZEROPAD])
	x[1][1] = char2idx[opt.tokens.START] -- start-of-word symbol
	local i = 2
	for char_code, char in pairs(UTF8ToCharArray(word)) do
		 if char2idx[char]==nil then -- unknown char
			 x[1][i] = char2idx['н']
		 else	-- known char
			 x[1][i] = char2idx[char]
		 end
		 i = i + 1
	end
	x[1][i] = char2idx[opt.tokens.END] -- end-of-word symbol
	return x:cuda(), y
end

-- do fwd/bwd and return loss, grad_params
local text = "служебного быта и соииокульт^'рной деятельности ;"
function speak(text)

	local words = split2words(text)
	local x, y = {}, {}
	for i = 1, #words do
		x[#x + 1], y[#y + 1] = word2x(words[i])
	end

	local state = clone_list(init_state)
	local predictions = {}					 -- softmax outputs
	local loss = 0

	local text = ""
	local decoded = ""
	local input = ""
	for t = 1, #words do

		local lst = protos.rnn:forward({x[t],unpack(state)})
		state = {}
		for i=1,#init_state do table.insert(state, lst[i]) end
		local prediction = lst[#lst]

		--local singleton_loss = protos.criterion:forward(prediction, y[{{1,2},t}])
		--loss = loss + singleton_loss

		-- introspect

		local loss, index = torch.max(prediction[1],1)
		--print(loss, index)
		decoded = decoded .. " " .. idx2word[index[1]]
		text = text .. " " .. idx2word[y[t]]
		input = input .. " "
		for ch = 2, x[t]:size(2) do
			local char = idx2char[x[t][1][ch]]
			if char~= '}' then
				input = input .. idx2char[x[t][1][ch]]
			else
				break
			end
		end

	end

	print("input  ->",input)
	print("true   ->",text)
	print("decoded<-",decoded)
	--loss = loss / opt.seq_length
end

while true do
  text = io.read('*l')
	speak(text)
end
