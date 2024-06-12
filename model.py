import sys

import torch
import torch.nn as nn
from pyod.models.ecod import ECOD
from baseCDM.NCD import NCD
from baseCDM.KANCD import KANCD
from baseCDM.DINA import DINA
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    '''
    NeuralCDM
    '''

    def __init__(self, student_n, exer_n, knowledge_n, time_graph):
        super(Net, self).__init__()
        self.time_graph = time_graph
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.boundaries = torch.tensor([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], device=device)
        self.time_embedding = nn.Embedding(self.boundaries.size(0) + 1, 128)
        self.time_effect_embedding = nn.Embedding(self.boundaries.size(0) + 1, 128)

        self.multihead_attn_a = nn.MultiheadAttention(128, 2, batch_first=True)
        self.multihead_attn_b = nn.MultiheadAttention(128, 2, batch_first=True)
        self.self_multihead_attn = nn.MultiheadAttention(128, 1, batch_first=True)

        self.hint_embedding = nn.Embedding(self.knowledge_dim, 128)
        self.hintBN = nn.BatchNorm1d(128)
        self.hint_FC = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 128)
        )
        self.FC = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

        # network structure
        self.baseCDM_type = sys.argv[3]
        print(f"use {self.baseCDM_type} model")
        assert self.baseCDM_type in ['NCD', "KANCD", "DINA"]
        if self.baseCDM_type == 'NCD':
            self.baseCDM = NCD(self.student_n, self.exer_n, self.knowledge_dim)
        if self.baseCDM_type == 'KANCD':
            self.baseCDM = KANCD(self.student_n, self.exer_n, self.knowledge_dim)
        if self.baseCDM_type == 'DINA':
            self.baseCDM = DINA(self.student_n, self.exer_n, self.knowledge_dim)

        self.add_or_not = sys.argv[4] == "add"
        if self.add_or_not:
            print("add my additional framework")
        else:
            print("no additional charges")
        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name and "BN" not in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb, time_taken, skill_index):
        output = None

        if self.baseCDM_type == 'NCD':
            output = self.baseCDM(stu_id, exer_id, kn_emb)
        if self.baseCDM_type == 'KANCD':
            output = self.baseCDM(stu_id, exer_id, kn_emb)
        if self.baseCDM_type == 'DINA':
            output = self.baseCDM(stu_id, exer_id, kn_emb)

        if self.add_or_not:
            attn_output_a = []

            for index, item in enumerate(stu_id):
                a, i = self.time_graph.get_all_problem_time(item, time_taken[index])
                a_embed = self.time_embedding(torch.bucketize(a.to(device), self.boundaries))
                a = torch.unsqueeze(a, dim=1)
                all_AD_result = ECOD().fit(a).decision_scores_
                AD_result = all_AD_result[i]
                all_AD_result = torch.tensor(all_AD_result, device=device)
                AD_result = torch.tensor(AD_result, device=device)
                weight = -torch.pow((all_AD_result - AD_result), 2)
                weight = torch.softmax(weight, dim=0)
                weight = torch.unsqueeze(weight, dim=1)
                result = torch.mul(weight, a_embed)
                result = torch.sum(result, dim=0)
                result = torch.squeeze(result, dim=0)
                attn_output_a.append(result)
            attn_output_a = torch.stack(attn_output_a)
            attn_output_a = torch.unsqueeze(attn_output_a, dim=1).float()

            attn_output_b = []

            for index, item in enumerate(exer_id):
                a, i = self.time_graph.get_all_student_time(item, time_taken[index])
                a_embed = self.time_embedding(torch.bucketize(a.to(device), self.boundaries))
                a = torch.unsqueeze(a, dim=1)
                all_AD_result = ECOD().fit(a).decision_scores_
                AD_result = all_AD_result[i]
                all_AD_result = torch.tensor(all_AD_result, device=device)
                AD_result = torch.tensor(AD_result, device=device)
                weight = -torch.pow((all_AD_result - AD_result), 2)
                weight = torch.softmax(weight, dim=0)
                weight = torch.unsqueeze(weight, dim=1)
                result = torch.mul(weight, a_embed)
                result = torch.sum(result, dim=0)
                result = torch.squeeze(result, dim=0)
                attn_output_b.append(result)
            attn_output_b = torch.stack(attn_output_b)
            attn_output_b = torch.unsqueeze(attn_output_b, dim=1).float()

            hint_embeding = self.hint_embedding(skill_index)
            hint_embeding_new = self.hintBN(hint_embeding)
            hint_embeding_new = self.hint_FC(hint_embeding_new)
            hint_embeding_new = hint_embeding_new + hint_embeding
            hint_embeding_new = torch.unsqueeze(hint_embeding_new, dim=1)
            cat_data = torch.concat([attn_output_a, attn_output_b, hint_embeding_new], dim=1)
            self_multihead_attn_output, self_multihead_attn_output_w = self.self_multihead_attn(cat_data, cat_data,
                                                                                                cat_data)
            self_multihead_attn_output = torch.reshape(self_multihead_attn_output,
                                                       (self_multihead_attn_output.size(0), -1))
            finally_data = self.FC(self_multihead_attn_output)

            output = output * finally_data[:, 0] + (1 - output) * finally_data[:, 1]
        return output

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data
