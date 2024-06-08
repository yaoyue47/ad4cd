import json

import pandas as pd
from sklearn.model_selection import train_test_split


def drop_null(a):
    b = a.dropna()
    print(f"去除null成功，删除{len(a) - len(b)}行")
    return b


def drop_duplicate(a):
    b = a.drop_duplicates()
    print(f"去除重复成功，删除{len(a) - len(b)}行")
    return b


def delete_little(a, columns_name, num):  # a:输入的frame，columns_name：要检测数量的列，num：最小值
    b = a.groupby(columns_name).count()
    c = b[b.columns[0]].rename("times")
    d = pd.merge(a, c, how="left", left_on=columns_name, right_index=True)
    e = d.drop(d[d['times'] <= num].index)
    f = e.drop(columns="times")
    print(f"检测{columns_name}字段成功,共删除{len(a) - len(f)}行")
    return f


def re_arrange(a, columns_name):  # a:输入的frame，columns_name，要重新从0排列的字段
    b = a[columns_name].drop_duplicates(ignore_index=True)
    print(f"重新排列成功，{columns_name}现在有{len(b)}行")
    c = b.reset_index()
    d = pd.merge(a, c, how="left")
    e = d.drop(columns=columns_name).rename(columns={"index": columns_name})
    return e


def save_json(a):
    dict = {
        "all_n": len(a),
        "problem_n": (a["problem_id"].max()) + 1,
        "skill_n": (a["skill_id"].max()) + 1,
        "user_n": (a["user_id"].max()) + 1
    }
    json_str = json.dumps(dict, indent=4)
    with open('./num.json', 'w') as json_file:
        json_file.write(json_str)


def get_test_and_kfold_data(input_data, columns_name):  # a:输入的frame，columns_name，给我学生id的字段
    def divide_data(data, scale):
        added_data = data[columns_name]
        data_L, data_R, useless_0, useless_1 = train_test_split(data, added_data, stratify=added_data, test_size=scale)
        return data_L, data_R

    print(f"分割前共有{len(input_data)}行")
    a, test = divide_data(input_data, 0.2)
    b, train_0 = divide_data(a, 0.2)
    c, train_1 = divide_data(b, 0.25)
    d, train_2 = divide_data(c, 1 / 3)
    train_4, train_3 = divide_data(d, 0.5)
    print(f"分割后{len(test)}\n{len(train_0)}\n{len(train_1)}\n{len(train_2)}\n{len(train_3)}\n{len(train_4)}\n")
    return test, train_0, train_1, train_2, train_3, train_4,


if __name__ == '__main__':
    src = ""  # 修改这里
    a = pd.read_csv(src)
    a = drop_null(a)
    a = drop_duplicate(a)
    # a = delete_little(a, "problem_id", 10)
    a = delete_little(a, "user_id", 10)
    a = re_arrange(a, "user_id")
    a = re_arrange(a, "problem_id")
    a = re_arrange(a, "skill_id")
    save_json(a)
    a.to_csv(src, index=False)
    test, train_0, train_1, train_2, train_3, train_4 = get_test_and_kfold_data(a, "user_id")
    test.to_csv("./test.csv", index=False)
    train_0.to_csv("./train_0.csv", index=False)
    train_1.to_csv("./train_1.csv", index=False)
    train_2.to_csv("./train_2.csv", index=False)
    train_3.to_csv("./train_3.csv", index=False)
    train_4.to_csv("./train_4.csv", index=False)
    print("保存文件成功")
