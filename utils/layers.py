import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.ModuleList) :
    def __init__(self, input_size, hidden_size, bias=True, bias_init=0.0):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size, bias=bias)
        if bias_init is not None and bias:
            self.linear.bias.data.fill_(bias_init)
        elif bias:
            self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        out = self.linear(x)
        return out

class Summ(nn.Module) :
    def __init__(self, input_size, dropout):
        super(Summ, self).__init__()
        self.dropout = dropout
        self.w = Linear(input_size, 1)

    def forward(self, x, mask) :
        d_x = Dropout(x, self.dropout, self.training)
        beta = self.w(d_x).squeeze(2)
        beta = beta + mask
        beta = F.softmax(beta, 1)
        output = torch.bmm(beta.unsqueeze(1), x).squeeze(1)
        return output

class Fusion(nn.Module) :
    def __init__(self, input_size, dropout):
        super(Fusion, self).__init__()
        self.input_size = input_size
        self.dropout = dropout

        self.W_r = Linear(4*input_size, input_size)
        self.W_g = Linear(4*input_size, input_size)

    def forward(self, x, y) :
        interact_data = torch.cat([x, y, x * y, x - y], dim=2)
        d_interact_data = Dropout(interact_data, self.dropout, self.training)
        x_tilde = gelu(self.W_r(d_interact_data))
        d_interact_data = Dropout(interact_data, self.dropout, self.training)
        g = torch.sigmoid(self.W_g(d_interact_data))
        output = g * x_tilde + (1. - g) * x
        return output

class MnemonicPointerNet(nn.Module) :
    def __init__(self, input_size, dropout):
        super(MnemonicPointerNet, self).__init__()
        self.dropout = dropout

        self.W_s = Linear(4*input_size, input_size)
        self.w_s = Linear(input_size, 1)
        self.W_e = Linear(4*input_size, input_size)
        self.w_e = Linear(input_size, 1)
        self.fusion_layer = Fusion(input_size, dropout)

    def forward(self, self_states, p_mask, init_states):
        batch_size, P, _ = self_states.size()
        s = init_states.unsqueeze(1).repeat(1, P, 1)
        concat_start_data = torch.cat([self_states, s, self_states * s, self_states - s], dim=2)
        d_concat_start_data = Dropout(concat_start_data, self.dropout, self.training)
        Ws = torch.tanh(self.W_s(d_concat_start_data))
        d_Ws = Dropout(Ws, self.dropout, self.training)
        logits1 = self.w_s(d_Ws).squeeze(2)
        logits1 = logits1 + p_mask

        P_s = F.softmax(logits1, 1)
        L = torch.bmm(P_s.unsqueeze(1), self_states).expand_as(self_states)

        s_tilde = self.fusion_layer(L, s)

        concat_end_data = torch.cat([self_states, s_tilde, self_states * s_tilde, self_states - s_tilde], dim=2)

        d_concat_end_data = Dropout(concat_end_data, self.dropout, self.training)

        We = torch.tanh(self.W_e(d_concat_end_data))

        d_We = Dropout(We, self.dropout, self.training)

        logits2 = self.w_e(d_We).squeeze(2)
        logits2 = logits2 + p_mask

        return logits1, logits2

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def Dropout(x, dropout, is_training, return_mask = False, var=False):
    if not var and not return_mask:
        return F.dropout(x, dropout, is_training)

    if dropout > 0.0 and is_training:
        shape = x.size()
        keep_prob = 1.0 - dropout
        random_tensor = keep_prob
        if var:
            tmp = torch.FloatTensor(shape[0], 1, shape[2])
        else:
            tmp = torch.FloatTensor(shape[0], shape[1], shape[2])

        tmp = tmp.to(x.device)
        nn.init.uniform_(tmp)
        random_tensor += tmp
        binary_tensor = torch.floor(random_tensor)
        x = torch.div(x, keep_prob) * binary_tensor

    if return_mask:
        return binary_tensor

    return x