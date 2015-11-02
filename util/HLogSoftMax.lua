local HLogSoftMax, parent = torch.class('nn.HLogSoftMax', 'nn.Criterion')

function HLogSoftMax:__init(mapping, input_size)
    -- different implementation of the fbnn.HSM module
    -- variable names are mostly the same as in fbnn.HSM
    -- only supports batch inputs

    parent.__init(self)
    if type(mapping) == 'table' then
         self.mapping = torch.LongTensor(mapping)
    else
         self.mapping = mapping
    end
    self.input_size = input_size
    self.n_classes = self.mapping:size(1)
    self.n_clusters = self.mapping[{{},1}]:max()
    self.n_class_in_cluster = torch.LongTensor(self.n_clusters):zero()
    for i = 1, self.mapping:size(1) do
        local c = self.mapping[i][1]
        self.n_class_in_cluster[c] = self.n_class_in_cluster[c] + 1
    end
    self.n_max_class_in_cluster = self.mapping[{{},2}]:max()

    --cluster softmax/loss
    self.cluster_model = nn.Sequential()
    self.cluster_model:add(nn.Linear(input_size, self.n_clusters))
		local lsm_cluster = nn.LogSoftMax()
		lsm_cluster.name = 'cluster_prediction'
		self.cluster_prediction = lsm_cluster
    self.cluster_model:add(lsm_cluster)
    --self.cluster_model:add(nn.LogSoftMax())
    self.logLossCluster = nn.ClassNLLCriterion()

    --class softmax/loss
    self.class_model = HSMClass.hsm(self.input_size, self.n_clusters, self.n_max_class_in_cluster)

		for _,node in ipairs(self.class_model.forwardnodes) do
			local name = node.data.annotations.name
			if name == 'class_prediction' then
				print('got class_prediction node')
				self.class_prediction = node.data.module
			elseif name == 'class_bias' then
				print('got class_bias node')
				self.class_bias = node.data.module
			elseif name == 'class_weight' then
				print('got class_weight node')
				self.class_weight = node.data.module
			end
		end

    self.logLossClass = nn.ClassNLLCriterion()

    self:change_bias()
    self.gradInput = torch.Tensor(input_size)
end

function HLogSoftMax:clone(...)
    return nn.Module.clone(self, ...)
end

function HLogSoftMax:parameters()
    return {self.cluster_model.modules[1].weight,
            self.cluster_model.modules[1].bias,
            self.class_bias.weight,
            self.class_weight.weight} ,
           {self.cluster_model.modules[1].gradWeight,
            self.cluster_model.modules[1].gradBias,
            self.class_bias.gradWeight,
            self.class_weight.gradWeight}
end

function HLogSoftMax:getParameters()
    return nn.Module.getParameters(self)
end

function HLogSoftMax:updateOutput(input, target)
    self.batch_size = input:size(1)
    local new_target = self.mapping:index(1, target)
    local cluster_loss = self.logLossCluster:forward(
                   self.cluster_model:forward(input),
                   new_target:select(2,1))
    local class_loss = self.logLossClass:forward(
                        self.class_model:forward({input, new_target:select(2,1)}),
                        new_target:select(2,2))
    self.output = cluster_loss + class_loss
    return self.output
end

function HLogSoftMax:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    local new_target = self.mapping:index(1, target)
    -- backprop clusters
    self.logLossCluster:updateGradInput(self.cluster_model.output,
                                        new_target:select(2,1))
    self.gradInput:copy(self.cluster_model:backward(input,
                        self.logLossCluster.gradInput))
    -- backprop classes
    self.logLossClass:updateGradInput(self.class_model.output,
                                      new_target:select(2,2))
    self.gradInput:add(self.class_model:backward(input,
                       self.logLossClass.gradInput)[1])
    return self.gradInput
end


function HLogSoftMax:backward(input, target, scale)
    self:updateGradInput(input, target)
    return self.gradInput
end

function HLogSoftMax:change_bias()
    -- hacky way to deal with variable cluster sizes
    for i = 1, self.n_clusters do
        local c = self.n_class_in_cluster[i]
        for j = c+1, self.n_max_class_in_cluster do
            self.class_bias.weight[i][j] = math.log(0)
        end
    end
end
