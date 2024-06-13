import torch
import torch.nn as nn


class KSCD(nn.Module):

    def __init__(self, student_n, exer_n, knowledge_n, low_dim=20):
        super(KSCD, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.lowdim = low_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        # network structure
        self.student_emb = nn.Embedding(self.student_n, self.lowdim)  # 学生的低维表示
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.lowdim)  # 知识点矩阵的低维表示
        self.k_difficulty = nn.Embedding(self.exer_n, self.lowdim)  # 习题的低维表示

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        self.layer1 = nn.Linear(self.lowdim, 1)
        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        stu_low_emb = self.student_emb(stu_id)
        knowledge_low_emb = self.knowledge_emb(torch.arange(self.knowledge_dim).to(stu_id.device))
        stu_emb = torch.sigmoid(torch.mm(stu_low_emb, knowledge_low_emb.T))  # 得到表示学生能力

        # 习题难度表示
        exe_low_emb = self.k_difficulty(exer_id)
        k_difficulty = torch.sigmoid(torch.mm(exe_low_emb, knowledge_low_emb.T))  # 得到表示学生能力
        e_discrimination = torch.sigmoid(self.layer1(exe_low_emb)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        output = torch.squeeze(output, dim=1)
        return output
