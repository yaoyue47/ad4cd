import torch
from pandas import DataFrame


def getGraph(df: DataFrame, max_student, max_problem) -> torch.int:
    print("开始构建图")
    array = torch.full((max_student, max_problem), fill_value=-1)
    for i in df.itertuples(index=False):
        array[i.user_id][i.problem_id] = i.timeTaken
    print("完成构建图")
    return array
