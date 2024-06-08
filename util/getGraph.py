import torch
from pandas import DataFrame


class Graph:
    def __init__(self, df: DataFrame):
        df = df[["user_id", "problem_id", "timeTaken"]]
        array = torch.tensor(df.values)
        self.graph = df

    def get_all_problem(self, stu_id):

        pass

    def get_all_student(self, ):
        pass
