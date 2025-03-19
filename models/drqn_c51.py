import torch.nn as nn

class DRQN_C51(nn.Module):

	def __init__(self, state_size, action_size=3, num_atoms=51):
		super(DRQN_C51, self).__init__()
		self.num_atoms = num_atoms
		self.action_size = action_size
		self.first_two_layers = nn.Sequential(
			nn.Linear(state_size, 256),
			nn.ELU(),
			nn.Linear(256, 256),
			nn.ELU(),
		)
		# Added third linear layer
		self.third_layer = nn.Sequential(
			nn.Linear(256, 256),
			nn.ELU()
		)
		self.lstm = nn.LSTM(256, 256, 1, batch_first=True)
		self.last_linear = nn.Linear(256, action_size*num_atoms)

	def forward(self, input):
		x = self.first_two_layers(input)
		x = self.third_layer(x)
		lstm_out, hs = self.lstm(x)
		batch_size, seq_len, mid_dim = lstm_out.shape
		linear_in = lstm_out.contiguous().view(seq_len * batch_size, mid_dim)
		action_value_dist = self.last_linear(linear_in)
		action_value_dist = action_value_dist.view(batch_size*seq_len, self.action_size, self.num_atoms)
		action_value_dist = nn.functional.softmax(action_value_dist, dim=-1)
		return action_value_dist