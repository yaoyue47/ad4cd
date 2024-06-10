import torch
import torch.nn as nn


class NCD(nn.Module):
    '''
    NeuralCDM
    '''

    def __init__(self, student_n, exer_n, knowledge_n):
        super(NCD, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n

        # network structure
        self.student_emb = nn.Embedding(self.student_n, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):

        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb * 10
        input_x = torch.sum(input_x, 1)
        output = torch.sigmoid(input_x)

        return output
