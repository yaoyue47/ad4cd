import pandas as pd


def clear_fields(url, user_id_name, problem_id_name, skill_id_name, correct_name, timeTaken_name, hint_name):
    csv = pd.read_csv(url)
    all_fields = list(csv)
    keep_fields = [user_id_name, problem_id_name, skill_id_name, correct_name, timeTaken_name, hint_name]
    for i in keep_fields:
        all_fields.remove(i)
    csv.drop(
        columns=all_fields,
        inplace=True)
    update_fields = {
        user_id_name: "user_id",
        problem_id_name: "problem_id",
        skill_id_name: "skill_id",
        correct_name: "correct",
        timeTaken_name: "timeTaken",
        hint_name: "hint"
    }
    csv.rename(columns=update_fields, inplace=True)
    csv.to_csv(url, index=False)
    print(
        f"完成对{user_id_name},{problem_id_name},{skill_id_name},{correct_name},{timeTaken_name},{hint_name}字段的修改")


if __name__ == '__main__':
    clear_fields()
