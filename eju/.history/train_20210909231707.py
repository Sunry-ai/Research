from enum import unique
import warnings
import pickle
import mlflow
import mlflow.sklearn
import logging
import copy
from statistics import mean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as classifier
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score

import lightgbm as lgb
from tqdm import tqdm
from make_feature import Feature_maker

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


model_save = "models_rft"


def lgb_model(X, y, X_test, y_test, name):
    # 敲定好一组参数
    params_ = {
        "num_leaves": 500,  # 结果对最终效果影响较大，越大值越好，太大会出现过拟合
        "min_data_in_leaf": 1,
        "objective": "binary",  # 定义的目标函数
        "max_depth": -1,
        "learning_rate": 0.11,
        "min_sum_hessian_in_leaf": 15,
        "boosting": "gbdt",
        "feature_fraction": 0.98,  # 提取的特征比率
        "bagging_freq": 1,
        "bagging_fraction": 0.95,
        "bagging_seed": 11,
        "lambda_l1": 0.011,  # l1正则
        "lambda_l2": 0.001,  # l2正则
        "verbosity": -1,
        "nthread": -1,  # 线程数量，-1表示全部线程，线程越多，运行的速度越快
        "metric": {"binary_logloss", "f1"},  # 评价函数选择
        "random_state": 2019,  # 随机数种子，可以防止每次运行的结果不一致
        # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
    }
    mlflow.log_param("params", params_)
    print("开始训练...")
    # 训练
    lgb_train = lgb.Dataset(X, y[y_para])
    lgb_eval = lgb.Dataset(X_test, y_test[y_para])
    gbm = lgb.train(
        params_,
        lgb_train,
        num_boost_round=1000,
        valid_sets=lgb_eval,
        early_stopping_rounds=250,
    )
    pickle.dump(gbm, open(model_save + "/" + name + ".joblib", "wb"))

def rft_mode(X, y, name):
    # ハイパパラメータを定義する
    parameters = {
        "max_depth": [i for i in [5, 10, 15]],
        "min_samples_leaf": [8, 10, 20],
        "random_state": [5],
    }
    # 森の数は200,random_stateを3に設定した
    forest = classifier(n_estimators=500, criterion="gini", random_state=3)
    # グリッドサーチして、交差検証を行うのを定義する
    # grid_search= sklearn.model_selection.GridSearchCV(forest, parameters,scoring="f1",cv=10,n_jobs=-1)
    # グリッドサーチして、交差検証を行う
    # grid_search.fit(X, y)
    # print(grid_search.best_score_)  # 最も良かったスコア
    # print(grid_search.best_params_)  # 上記を記録したパラメータの組み合わせ
    # その結果を可視化する
    # cv_result = pd.DataFrame(grid_search.cv_results_)
    # cv_result = cv_result[['param_max_depth', 'param_min_samples_leaf', 'mean_test_score']]
    # cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_max_depth', 'param_min_samples_leaf')
    # heat_map = sns.heatmap(cv_result_pivot, cmap="GnBu", annot=True);
    # plt.savefig(name+"svg")
    # 最も良い学習モデルで学習
    predictor = forest.fit(X, y)  # grid_search.best_estimator_
    # モデルを保存する
    pickle.dump(predictor, open(model_save + "/" + name + ".joblib", "wb"))

def final_result(item, th):
    if item > th:
        return 1
    else:
        return 0


def train(option="rft"):
    student_number = 150
    print("start train")
    remove_list=[]
    for school_id in tqdm(range(len(schoolname.keys()))):
        # -------------------------------------------------------测试集/训练集区分----------------------------------------------------------------------
        # df_original为单个学校的数据
        try:
            df_original = df_train.groupby("受験校").get_group(schoolname[school_id])[paras_all]
        except:
            #df_original = df_test.groupby("受験校").get_group(schoolname[school_id])[paras_all]
            remove_list.append(schoolname[school_id])
            continue
            

        # 如果测试数据集中没有该学校则使用原本数据集
        if len(df_test[df_test["受験校"] == schoolname[school_id]]) > 0:
            test = df_test[df_test["受験校"] == schoolname[school_id]][paras_all]
        else:
            #print(schoolname[school_id], "报名人数过少")
            remove_list.append(schoolname[school_id])
            continue

        # 为了克服数据不均衡问题，采用上采样策略
        # ------------------------------------------------------上采样--------------------------------------------------------------------------------

        # 如果合格人数为0，则判定为报名人中的最高分+20为合格的人的最高分数，否则，则以最高分数的人的分数为标准。
        num_1 = len(df_original[df_original["合否"] == 1])
        num_0 = len(df_original[df_original["合否"] == 0])

        if num_1 == 0:  # 如果没有人合格的话
            # 最高最低分
            p_max = df_original[df_original["合否"] == 0]["max(全部)"].max() + 5
            p_min = df_original[df_original["合否"] == 0]["max(全部)"].min()

        else:
            # 最高最低分
            p_max = df_original[df_original["合否"] == 1]["max(全部)"].max()
            p_min = df_original[df_original["合否"] == 1]["max(全部)"].min()

        # 总分低于【不合格人】的最低分的人判断为不合格
        # 总分低高于【合格人】的最高分的人判断为合格

        # 从整体数据集中对符合上述要求的学生进行上采样
        df_filter_0 = df[df["max(全部)"] < p_min]
        df_filter_1 = df[df["max(全部)"] > p_max]

        try:
            if num_0 < student_number:
                df_filter_0 = df_filter_0[paras_all].sample(n=student_number - num_0)
                df_filter_0["受験校"] = schoolname[school_id]
                df_filter_0["合否"] = 0
            else:
                df_tmp = df_original[df_original["合否"] == 0]
                df_filter_0 = df_tmp[paras_all].sample(n=num_1)
                df_filter_0["受験校"] = schoolname[school_id]
                df_filter_0["合否"] = 0
        except:
            df_filter_0 = df_original[df_original["合否"] == 0]

        try:
            df_filter_1 = df_filter_1[paras_all].sample(n=student_number - num_1)
        except:
            df_filter_1 = df_filter_1[paras_all]

        df_filter_1["受験校"] = schoolname[school_id]
        df_filter_1["合否"] = 1
        result = df_filter_1.append(df_filter_0)
        result = result.append(df_original)
        result.reset_index(drop=True, inplace=True)

        # ------------------------------------------------------训练&评价--------------------------------------------------------------------------------
        if option=="rft":
            try:
                with open(model_save + "/" + str(school_id) + ".joblib", mode="rb") as fp:
                    clf = pickle.load(fp)

            except:
                
                rft_mode(result[paras], result[y_para], str(school_id))

                with open(model_save + "/" + str(school_id) + ".joblib", mode="rb") as fp:
                    clf = pickle.load(fp)
        elif option=="lgb":
            try:
                with open(model_save + "/" + str(school_id) + ".joblib", mode="rb") as fp:
                    clf = pickle.load(fp)

            except:
                lgb_model(result[paras], result[y_para],test[paras],test[y_para], str(school_id))

                with open(model_save + "/" + str(school_id) + ".joblib", mode="rb") as fp:
                    clf = pickle.load(fp)

        # ----------------------------------------------训练数据集-----------------------------------------------------------------------------------------
        result_model = list(clf.predict(df_original[paras]))

        real = list(df_original["合否"])
        count = 0

        result_model = [final_result(item, 0.5) for item in result_model]

        for i in range(len(real)):
            if real[i] == result_model[i]:
                count += 1
            else:
                pass

        accuracy = count / len(real)

        recall = recall_score(result_model, real)
        precistion = precision_score(result_model, real)
        try:
            f1 = f1_score(result_model, real)
        except:
            f1 = 0.5

        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precistion)
        f1_list.append(f1)

        # ----------------------------------------------测试数据集----------------------------------------------------------------------------------
        test_model = list(clf.predict(test[paras]))
        test_model = [final_result(item, 0.5) for item in test_model]
        real = list(test["合否"])

        count = 0
        for i in range(len(real)):
            if real[i] == test_model[i]:
                count += 1
            else:
                pass

        test_accuracy = count / len(real)
        test_recall = recall_score(test_model, real)
        test_precistion = precision_score(test_model, real)
        try:
            test_f1 = f1_score(test_model, real)
        except:
            test_f1 = 0.5

        accuracy_list_test.append(test_accuracy)
        print(accuracy_list.mean())
        recall_list_test.append(test_recall)
        precision_list_test.append(test_precistion)
        f1_list_test.append(test_f1)

    # ------------------------------------------------------结果展示--------------------------------------------------------------------------------
    print("test result is",mean(test_accuracy),mean(test_recall),mean(test_precistion),mean(test_f1))
    # Train_result = pd.DataFrame(
    #     data={
    #         "学校名": list(schoolname.values()),
    #         "精度": accuracy_list,
    #         "召回率": recall_list,
    #         "適合率": precision_list,
    #         "f1": f1_list,
    #     }
    # )
    # Test_result = pd.DataFrame(
    #     data={
    #         "学校名": list(schoolname.values()),
    #         "精度": accuracy_list_test,
    #         "召回率": recall_list_test,
    #         "適合率": precision_list_test,
    #         "f1": f1_list_test,
    #     }
    # )

    # def judeg_in_remove(x):
    #     if x in remove_list:
    #         return "NO"
    #     else:
    #         return x
    # Test_result.学校名=Test_result.学校名.apply(judeg_in_remove)
    # Test_result=Test_result[Test_result.学校名!="NO"]

    # print("remove list length is" ,len(remove_list))

    # mlflow.log_metric("accuracy", Train_result["精度"].mean())
    # mlflow.log_metric("recall", Train_result["召回率"].mean())
    # mlflow.log_metric("precistion", Train_result["適合率"].mean())
    # mlflow.log_metric("f1", Train_result["f1"].mean())

    # Test_result = Test_result[Test_result["精度"] != 0]
    # Test_result = Test_result[Test_result["召回率"] != 0]
    # Test_result = Test_result[Test_result["適合率"] != 0]
    # Test_result = Test_result[Test_result["f1"] != 0]

    # mlflow.log_metric("test accuracy", Test_result[Test_result["精度"] != 0]["精度"].mean())
    # mlflow.log_metric("test recall", Test_result[Test_result["召回率"] != 0]["召回率"].mean())
    # mlflow.log_metric(
    #     "test precistion", Test_result[Test_result["適合率"] != 0]["適合率"].mean()
    # )
    # mlflow.log_metric("test f1", Test_result[Test_result["f1"] != 0]["f1"].mean())
    # mlflow.log_param("features", str(paras))


if __name__ == "__main__":

    # データ取得
    feature_maker = Feature_maker("./dataset/data_raw2021_V1.xlsx")
    paras, schoolname, df = feature_maker.main()
    # Params設定

    paras_all = copy.deepcopy(paras)
    paras_all.append("合否")
    paras_all.append("受験校")
    paras_all.append("年度")
    y_para = ["合否"]
    # Train / Test
    
    df_test = df[df["年度"] == 2021][paras_all]
    df_train = df[df["年度"] != 2021][paras_all]

    school = schoolname
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_list = []

    accuracy_list_test = []
    recall_list_test = []
    precision_list_test = []
    f1_list_test = []

    with mlflow.start_run(run_name="训练与测试数据集"):
        mlflow.log_param("model", "rft")
        train(option="rft")
