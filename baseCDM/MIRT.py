import torch
import torch.nn as nn


class MIRT(nn.Module):

    def __init__(self, student_n, exer_n):
        super(MIRT, self).__init__()
        self.num_dim = 2
        self._alpha_params = nn.Embedding(exer_n, self.num_dim)  # self.num_dim表示嵌入维度
        self._beta_params = nn.Embedding(exer_n, 1)
        self._theta_params = nn.Embedding(student_n, self.num_dim)
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(p)
            if '_alpha_params' in name:
                nn.init.uniform_(p, a=0.5, b=2.5)

    def forward(self, stu_id, exer_id):

        theta = self._theta_params(stu_id)
        alpha = self._alpha_params(exer_id)
        beta = self._beta_params(exer_id)
        pred = (alpha * (theta - beta)).sum(dim=1, keepdim=True)
        output = torch.sigmoid(pred)
        output = torch.squeeze(output, dim=1)

        return output
