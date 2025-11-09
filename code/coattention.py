import torch
import torch.nn as nn
import torch.nn.functional as F


class CoAttention(nn.Module):
    '''
    外部知识融入模块
    '''
    def __init__(self, hidde_size: int, attention_size: int):  # hidden_size:d, attention_size:k
        super(CoAttention, self).__init__()

        self.hidden_size = hidde_size
        self.Wl = nn.Parameter(torch.zeros(size=(hidde_size * 2, hidde_size * 2)), requires_grad=True)
        self.Ws = nn.Parameter(torch.zeros(size=(attention_size, hidde_size * 2)), requires_grad=True)
        self.Wc = nn.Parameter(torch.zeros(size=(attention_size, hidde_size * 2)), requires_grad=True)
        self.whs = nn.Parameter(torch.zeros(size=(1, attention_size)), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.Wl.data.uniform_(-1.0, 1.0)
        self.Ws.data.uniform_(-1.0, 1.0)
        self.Wc.data.uniform_(-1.0, 1.0)
        self.whs.data.uniform_(-1.0, 1.0)

    def forward(self, new_batch, entity_desc_batch):
  
        S = torch.transpose(new_batch, 1, 2)
 
        C = torch.transpose(entity_desc_batch, 1, 2)
    
        attF = torch.tanh(torch.bmm(torch.transpose(C, 1, 2), torch.matmul(self.Wl, S)))

        WsS = torch.matmul(self.Ws, S)  
 
        WsC = torch.matmul(self.Wc, C) 
  
        Hs = torch.tanh(WsS + torch.bmm(WsC, attF))  # dim[batch,a,N]

        a_s = F.softmax(torch.matmul(self.whs, Hs), dim=2)  # dim[batch,1,N]

        s = torch.bmm(a_s, new_batch).squeeze()  # dim[batch,2h]

        return s
    
def split_s(a, n):  
    k, m = divmod(len(a), n)  
    
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))