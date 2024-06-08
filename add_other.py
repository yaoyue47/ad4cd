import pandas as pd
from scipy.stats import entropy


# def getEntropy(problem_id):
#     a1 = a[a["problem_id"] == problem_id]["timeTaken"]
#     b1 = entropy(pk=a1.value_counts(), base=2)
#     b1 = b1 * 100
#     return int(b1)
#
#
# def getMedian(problem_id):
#     a1 = a[a["problem_id"] == problem_id]["timeTaken"]
#     b1 = a1.median()
#     return int(b1)


def addOther(x, mean, std):
    # x["median"] = getMedian(x["problem_id"])
    # x["entropy"] = getEntropy(x["problem_id"])
    x["timeTaken"] = (x["timeTaken"]-mean)/std
    return x


srcArr = ["data/ASSIST2009/test.csv", "data/ASSIST2009/train.csv", "data/ASSIST2009/valid.csv"]  # 修改此处即可
# for src in srcArr:
#     a = pd.read_csv(src)
#     mean = a["timeTaken"].mean()
#     std = a["timeTaken"].std()
#     a["timeTaken"] = (a["timeTaken"]-mean)/std
#     # a = a.apply(addOther, axis=1, mean=mean, std=std)
#     print(a)
#     a.to_csv(src, index=False)
#     print(f"已修改{src}")

