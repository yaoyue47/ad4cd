import torch
import torch.nn as nn

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
        self.student_emb = nn.Embedding(self.student_n, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name and "BN" not in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb, time_taken, skill_index):
        a = torch.index_select(self.time_graph, 0, stu_id)
        b = torch.index_select(self.time_graph, 1, exer_id)

        a = torch.bucketize(a, self.boundaries)
        b = torch.bucketize(b, self.boundaries)
        b = torch.transpose(b, dim0=0, dim1=1)
        time_taken = torch.bucketize(time_taken, self.boundaries)

        a_time = self.time_embedding(a)
        b_time = self.time_embedding(b)
        time_taken_time = self.time_embedding(time_taken)
        time_taken_time = torch.unsqueeze(time_taken_time, dim=1)

        a_effect_time = self.time_effect_embedding(a)
        b_effect_time = self.time_effect_embedding(b)

        attn_output_a, attn_output_weights_a = self.multihead_attn_a(time_taken_time, a_time, a_effect_time)
        attn_output_b, attn_output_weights_b = self.multihead_attn_b(time_taken_time, b_time, b_effect_time)
        hint_embeding = self.hint_embedding(skill_index)
        hint_embeding_new = self.hintBN(hint_embeding)
        hint_embeding_new = self.hint_FC(hint_embeding_new)
        hint_embeding_new = hint_embeding_new + hint_embeding
        hint_embeding_new = torch.unsqueeze(hint_embeding_new, dim=1)
        cat_data = torch.concat([attn_output_a, attn_output_b, hint_embeding_new], dim=1)
        self_multihead_attn_output, self_multihead_attn_output_w = self.self_multihead_attn(cat_data, cat_data,
                                                                                            cat_data)
        self_multihead_attn_output = torch.reshape(self_multihead_attn_output, (self_multihead_attn_output.size(0), -1))
        finally_data = self.FC(self_multihead_attn_output)

        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))

        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb * 10
        input_x = torch.sum(input_x, 1)
        output = torch.sigmoid(input_x)
        output = output * finally_data[:, 0] + (1 - output) * finally_data[:, 1]
        return output

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data
