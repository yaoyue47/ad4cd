import torch
import torch.nn as nn
import numpy as np


class DINA(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, max_slip=0.4, max_guess=0.4):
        super(DINA, self).__init__()
        self._user_num = student_n
        self._item_num = exer_n
        self.knowledge_dim = knowledge_n
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess

        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, self.knowledge_dim)

    def forward(self, user, item, knowledge):
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)

        n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
        t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                           1e-6), self.step + 1 if self.step < self.max_step else 0
        output = torch.sum(
            torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
            dim=1
        )

        return output
