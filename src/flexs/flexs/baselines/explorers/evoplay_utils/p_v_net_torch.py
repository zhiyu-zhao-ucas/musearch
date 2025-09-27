# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height, alphabet):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv1d(len(alphabet), 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        # action policy layers
        self.act_conv1 = nn.Conv1d(128, 4*len(alphabet), kernel_size=5, padding=2)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv1d(128, len(alphabet), kernel_size=5, padding=2)
        self.val_fc1 = nn.Linear(board_width*board_height, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        # print(f"p_v_net_torch.py 43 x shape: {x.shape}")
        x_act = F.relu(self.act_conv1(x))
        # print(f"p_v_net_torch.py 45 x_act shape: {x_act.shape}, board_width: {self.board_width}, board_height: {self.board_height}, 4*board_width*board_height: {4*self.board_width*self.board_height}")
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        # print(f"p_v_net_torch.py 48 x_act shape: {x_act.shape}")
        x_val = F.relu(self.val_conv1(x))
        # print(f"p_v_net_torch.py 50 x_val shape: {x_val.shape}")
        x_val = x_val.view(-1, self.board_width*self.board_height)
        # print(f"p_v_net_torch.py 52 x_val shape: {x_val.shape}")
        x_val = F.relu(self.val_fc1(x_val))
        # print(f"p_v_net_torch.py 54 x_val shape: {x_val.shape}")
        x_val = torch.tanh(self.val_fc2(x_val))
        # print(f"p_v_net_torch.py 56 x_val shape: {x_val.shape}")
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True, alphabet=None):
        self.requested_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.device = torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            selected_device = torch.device("cuda")
            supported_arches = []
            try:
                supported_arches = torch.cuda.get_arch_list()
            except AttributeError:
                supported_arches = []

            arch = None
            try:
                current_device = torch.cuda.current_device()
                capability = torch.cuda.get_device_capability(current_device)
                arch = f"sm_{capability[0]}{capability[1]}"
            except Exception:
                capability = None

            supported = {a.lower() for a in supported_arches}
            arch_key = arch.lower() if arch else None
            compute_key = None
            if arch_key and arch_key.startswith("sm_"):
                compute_key = arch_key.replace("sm_", "compute_")

            arch_supported = False
            if arch_key:
                arch_supported = (not supported) or (arch_key in supported) or (compute_key in supported)

            if arch_supported:
                self.device = selected_device
            else:
                device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
                supported_str = ", ".join(supported_arches) if supported_arches else "unknown"
                warnings.warn(
                    (
                        f"Falling back to CPU: CUDA capability {arch or 'unknown'} of device '{device_name}' "
                        f"is not supported by this PyTorch build (supported architectures: {supported_str})."
                    ),
                    RuntimeWarning,
                )

        self.use_gpu = self.device.type == "cuda"
        # the policy value net module
        self.policy_value_net = Net(board_width, board_height, alphabet).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch_tensor = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch_tensor)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        return act_probs, value.detach().cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state_0 = np.expand_dims(board.current_state(), axis = 0)
        current_state = np.ascontiguousarray(current_state_0)  ##

        # print(f"p_v_net_torch.py 100 current_state: {current_state}")
        state_tensor = torch.from_numpy(current_state).float().to(self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.detach().cpu().numpy()[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        state_batch = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        mcts_probs = torch.as_tensor(mcts_probs, dtype=torch.float32, device=self.device)
        winner_array = np.asarray(winner_batch, dtype=np.float32)
        winner_batch_tensor = torch.from_numpy(winner_array).to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # print(f"p_v_net_torch.py 135 state_batch shape: {state_batch.shape}, value shape: {value.shape}, winner_batch shape: {winner_batch_tensor.shape}")
        value_loss = F.mse_loss(value.view(-1), winner_batch_tensor)
        # print(f"p_v_net_torch.py 143 mcts_probs shape: {mcts_probs.shape}, log_act_probs shape: {log_act_probs.shape}")
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )

        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
