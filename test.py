import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from data_loader import MyDataSet, dataset
from torch.utils.data import DataLoader
from util.pb4dl import pb4dl
from loss_function import MyLossFunction
loss_function = MyLossFunction()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
testDataSet = MyDataSet("test")
testDataLoader = DataLoader(testDataSet, 256, pin_memory=True, num_workers=4)


def test(epoch):
    net = torch.load("model/" + dataset + f"/epoch{epoch}.pth").to(device)
    net.eval()
    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    with torch.no_grad():
        for i in pb4dl(testDataLoader):
            studentId, problemId, skill_num, labels, time_taken, hint, skill_index = i
            studentId, problemId, skill_num, labels, time_taken, hint, skill_index = studentId.to(device), problemId.to(
                device), skill_num.to(device), labels.to(device), time_taken.to(device), hint.to(
                device), skill_index.to(device)
            output, new_E = net.forward(studentId, problemId, skill_num, time_taken, hint, skill_index)

            labels = labels.float()
            for a in range(len(labels)):
                if (labels[a] == 1 and output[a] > 0.5) or (labels[a] == 0 and output[a] < 0.5):
                    correct_count += 1
            exer_count += len(labels)
            pred_all += output.tolist()
            label_all += labels.tolist()

        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        accuracy = correct_count / exer_count
        mse = mean_squared_error(label_all, pred_all)
        rmse = np.sqrt(mse)
        auc = roc_auc_score(label_all, pred_all)
        print('accuracy= %f, mse= %f, rmse= %f, auc= %f' % (accuracy, mse, rmse, auc))


if __name__ == '__main__':
    epoch = sys.argv[2]
    test(epoch)
