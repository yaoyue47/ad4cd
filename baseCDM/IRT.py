import torch
import torch.nn as nn


class IRT(nn.Module):

    def __init__(self, student_n, exer_n):
        super(IRT, self).__init__()
        self.exer_n = exer_n
        self.student_n = student_n

        # network structure
        self.student_emb = nn.Embedding(self.student_n, 1)
        self.k_difficulty = nn.Embedding(self.exer_n, 1)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id):
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))

        output = 1 / (1 + torch.exp(-e_discrimination * 1.7 * (stu_emb - k_difficulty)))
        output = torch.squeeze(output, dim=1)

        return output
