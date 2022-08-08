import warnings 
import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as classifier
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import joblib
from tqdm import tqdm
import pickle
import mlflow
import mlflow.sklearn
import logging
import lightgbm as lgb
# 构建lgb中的Dataset格式

warnings.simplefilter('ignore')
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



model_save="models_lgb"
#交差検証用の関数
def rft_mode(X,y,name):
    #ハイパパラメータを定義する
    parameters = {
        "max_depth":[i for i in [5,10,15]],
        'min_samples_leaf': [8,10,20],
        "random_state":[5]
    }
    #森の数は200,random_stateを3に設定した
    forest=classifier(n_estimators=500,criterion="entropy",random_state=3)
    #グリッドサーチして、交差検証を行うのを定義する
    grid_search= sklearn.model_selection.GridSearchCV(forest, parameters,scoring="f1",cv=10,n_jobs=-1)
    #グリッドサーチして、交差検証を行う
    grid_search.fit(X, y)
    # print(grid_search.best_score_)  # 最も良かったスコア
    # print(grid_search.best_params_)  # 上記を記録したパラメータの組み合わせ
    #その結果を可視化する
    cv_result = pd.DataFrame(grid_search.cv_results_)
    cv_result = cv_result[['param_max_depth', 'param_min_samples_leaf', 'mean_test_score']]
    #cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_max_depth', 'param_min_samples_leaf')
    #heat_map = sns.heatmap(cv_result_pivot, cmap="GnBu", annot=True);
    #plt.savefig(name+"svg")
    #最も良い学習モデルで学習
    predictor=grid_search.best_estimator_
    #モデルを保存する
    pickle.dump(predictor,open(model_save+"/"+ name+'.joblib',"wb"))

def lgb_model(X,y,X_test,y_test,name):
    # 敲定好一组参数
    
    params = {'num_leaves': 10, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
            'min_data_in_leaf': 3,
            'objective': 'binary', #定义的目标函数
            'max_depth': -1,
            'learning_rate': 0.5,
            "min_sum_hessian_in_leaf": 15,
            "boosting": "gbdt",
            "feature_fraction": 0.95,  #提取的特征比率
            "bagging_freq": 1,
            "bagging_fraction": 0.9,
            "bagging_seed": 11,
            "lambda_l1": 0.1,             #l1正则
            'lambda_l2': 0.001,     #l2正则
            "verbosity": -1,
            "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
            'metric': {'binary_logloss', 'f1'},  ##评价函数选择
            "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
            # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
            }

    print('开始训练...')
    # 训练
    lgb_train = lgb.Dataset(X[x_para], y[y_para])
    lgb_eval = lgb.Dataset(X_test[x_para], y_test[y_para])
    gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=lgb_eval,
            early_stopping_rounds=25)
    pickle.dump(gbm,open(model_save+"/"+ name+'.joblib',"wb"))

def final_result(item,th):
        if item>th:
            return 1
        else:
            return 0
def train():
    student_number=100
    print("start train")
    for school_id in tqdm(range(len(school))):
    #-------------------------------------------------------测试集/训练集区分----------------------------------------------------------------------------------
        df_original=df_train.groupby("受験校").get_group(school[school_id])[para_2]
        #如果测试数据集中没有改学校则使用原本数据集
        if len(df_test[df_test["受験校"]==school[school_id]])>0:
            test=df_test[df_test["受験校"]==school[school_id]][para_2]
        else:
            break
            # test=df_original
            # print(school[school_id]+"没有测试集")
            
    #print(school[school_id],"报名人数过少")
    #为了克服数据不均衡问题，采用上采样策略
    #------------------------------------------------------上采样--------------------------------------------------------------------------------   
    
        #如果合格人数为0，则判定为报名人中的最高分+20为合格的人的最高分数，否则，则以最高分数的人的分数为标准。
        num_1=len(df_original[df_original["合否"]==1])
        num_0=len(df_original[df_original["合否"]==0])
        if num_1==0:
            p_max=df_original[df_original["合否"]==0]["max(全部)"].max()+20
            p_min=df_original[df_original["合否"]==0]["max(全部)"].min() 
        else:
            p_max=df_original[df_original["合否"]==1]["max(全部)"].max()
            p_min=df_original[df_original["合否"]==1]["max(全部)"].min()
        
        #总分低于【不合格人】的最低分的人判断为不合格
        #总分低高于【合格人】的最高分的人判断为合格
        
        #从整体数据集中对符合上述要求的学生进行上采样
        df_filter_0=df_all[df_all["max(全部)"]<p_min]
        df_filter_1=df_all[df_all["max(全部)"]>p_max]
        try:
            if num_0<student_number:
                df_filter_0=df_filter_0[para].sample(n=student_number-num_0)
                df_filter_0["受験校"]=school[school_id]
                df_filter_0["合否"]=0
            else:
                df_tmp=df_original[df_original["合否"]==0]
                df_filter_0=df_tmp[para].sample(n=num_1)
                df_filter_0["受験校"]=school[school_id]
                df_filter_0["合否"]=0
        except:
            df_filter_0=df_original[df_original["合否"]==0]
            
        try:
            df_filter_1=df_filter_1[para].sample(n=student_number-num_1)
        except:
            df_filter_1=df_filter_1[para]
        
        df_filter_1["受験校"]=school[school_id]
        df_filter_1["合否"]=1
        result=df_filter_1.append(df_filter_0)
        result=result.append(df_original)
        result.reset_index(drop=True, inplace=True)
        
        
    #---------------------------------------------训练&评价--------------------------------------------------------------------------------  
        try:
            with open(model_save+"/"+str(school_id)+'.joblib', mode='rb') as fp:
                clf = pickle.load(fp)
        except:
            lgb_model(result[x_para],result[y_para],test[x_para],test[y_para],str(school_id))
            with open(model_save+"/"+str(school_id)+'.joblib', mode='rb') as fp:
                clf = pickle.load(fp)
    #----------------------------------------------训练数据集----------------------------------------------
        result_model=list(clf.predict(result[x_para]))
        real=list(result["合否"])
        count=0
        result_model=[final_result(item,0.5) for item in result_model]
        for i in range(len(real)):
            if real[i]==result_model[i]:
                count+=1
            else:
                pass

        accuracy=count/len(real)
        recall=recall_score(result_model, real)
        precistion=precision_score(result_model, real)
        
        try:
            f1=f1_score(result_model,real)
        except:
            f1=0
        
        
        mlflow.log_metric("%s accuracy train"%schoolname[school_id], accuracy)
        mlflow.log_metric("%s recall train"%schoolname[school_id], recall)
        mlflow.log_metric("%s precistion train"%schoolname[school_id], precistion)
        mlflow.log_metric("%s f1 train"%schoolname[school_id], f1)
        
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precistion)
        f1_list.append(f1)
        
        #----------------------------------------------测试数据集----------------------------------------------
        test_model=list(clf.predict(test[x_para]))
        test_model=[final_result(item,0.5) for item in test_model]
        real=list(test["合否"])
        count=0
        for i in range(len(real)):
            if real[i]==test_model[i]:
                count+=1
            else:
                pass
        
        test_accuracy=count/len(real)
        test_recall=recall_score(test_model, real)
        test_precistion=precision_score(test_model, real)
        try:
            test_f1=f1_score(test_model,real)
        except:
            test_f1=0
        
        mlflow.log_metric("%s accuracy test"%schoolname[school_id], test_accuracy)
        mlflow.log_metric("%s recall test"%schoolname[school_id], test_recall)
        mlflow.log_metric("%s precistion test"%schoolname[school_id], test_precistion)
        mlflow.log_metric("%s f1 test"%schoolname[school_id], test_f1)
        print(feature_maker_test[schoolname[school_id]]["合格人数"])
        mlflow.log_metric("%s 合格数"%schoolname[school_id], feature_maker_test[schoolname[school_id]]["合格人数"])
        mlflow.log_metric("%s 受験者数"%schoolname[school_id], feature_maker_test[schoolname[school_id]]["受験人数"])
        
        accuracy_list_test.append(test_accuracy)
        recall_list_test.append(test_recall)
        precision_list_test.append(test_precistion)
        f1_list_test.append(test_f1)

    #------------------------------------------------------结果展示--------------------------------------------------------------------------------  
    Train_result=pd.DataFrame(data={"学校名":schoolname,"精度":accuracy_list,"召回率":recall_list,"適合率":precision_list,"f1":f1_list})
    Test_result=pd.DataFrame(data={"学校名":schoolname,"精度":accuracy_list_test,"召回率":recall_list_test,"適合率":precision_list_test,"f1":f1_list_test})
    mlflow.log_metric("accuracy",Train_result["精度"].mean())
    mlflow.log_metric("recall",Train_result["召回率"].mean())
    mlflow.log_metric("precistion",Train_result["適合率"].mean())
    mlflow.log_metric("f1",Train_result["f1"].mean())

    mlflow.log_metric("test accuracy",Test_result["精度"].mean())
    mlflow.log_metric("test recall",Test_result["召回率"].mean())
    mlflow.log_metric("test precistion",Test_result["適合率"].mean())
    mlflow.log_metric("test f1",Test_result["f1"].mean())
    mlflow.log_param("features", str(x_para))

if __name__ == "__main__":

    
    x_para=['日本語_', '記述', '物理_',
       '化学_', '生物_', '数2',"理科总分","理科综合", 'max(全部)',"物理数学","日语数学","化学数学"]

    para=['日本語_', '記述', '物理_',
        '化学_', '生物_', '数2',"理科总分","理科综合", 'max(全部)',
        "物理数学","日语数学","化学数学"]

    para_2=['受験校','日本語_', '記述', '物理_',
        '化学_', '生物_', '数2', 'max(全部)',"理科总分","理科综合","物理数学","日语数学","合否","化学数学"]

    y_para=["合否"]
    
    #Make Features
    print("start load")
    df_all=pd.read_excel("./dataset/data_raw.xlsx")

    print("Make Features")
    df_all["理科综合"]=df_all['物理_']+df_all['化学_']+df_all['生物_']
    df_all["理科总分"]=df_all['物理_']+df_all['化学_']+df_all['生物_']+df_all['数2']
    df_all["物理数学"]=df_all['物理_']+df_all['数2']
    df_all["日语数学"]=df_all['日本語_']+df_all['数2']
    df_all["化学数学"]=df_all['化学_']+df_all['数2']



    jap=[340,320,280,250]
    jw=[45,40,35,30]
    phy=[85,75,65,50]
    che=[85,75,65,50]
    bio=[85,75,65,50]
    math=[180,150,130,110]
    toeic=[800,700,600,400]
    toefl=[90,80,60,40]

    info=df_all[df_all["合否"]==1].groupby("受験校").mean()[para].sort_values("max(全部)")
    schoolname=list(info.index)
    schoolname.remove('千葉科学大学')
    schoolname.reverse()
    no=[]
    for i in range(len(schoolname)):
        no.append(str(i))
    id_dict=dict(zip(schoolname,no))
    inv_map = {v: k for k, v in id_dict.items()}
    df_test=df_all[df_all['年度']==2019]
    df_train=df_all[df_all['年度']!=2019]
    
    feature_maker={}
    feature_maker_test={}
    # for school in schoolname:
    #     df_tmp=df_train.groupby("受験校").get_group(school)
    
    #     pass_number_mean=df_tmp[df_tmp["合否"]==1].groupby(["年度"]).count().mean()["合否"]
    #     pass_number_std=df_tmp[df_tmp["合否"]==1].groupby(["年度"]).count().std()["合否"]
    #     pass_number_max=df_tmp[df_tmp["合否"]==1].groupby(["年度"]).count().max()["合否"]
    #     pass_number_min=df_tmp[df_tmp["合否"]==1].groupby(["年度"]).count().min()["合否"]
        
    #     all_number_mean=df_tmp.groupby(["年度"]).count().mean()["合否"]
    #     all_number_std=df_tmp.groupby(["年度"]).count().std()["合否"]
    #     all_number_max=df_tmp.groupby(["年度"]).count().max()["合否"]
    #     all_number_min=df_tmp.groupby(["年度"]).count().min()["合否"]
    
    #     dict_tmp={
    #         "合格平均人数":pass_number_mean,
    #         "合格人数分散":pass_number_std,
    #         "合格人数最大値":pass_number_max,
    #         "合格人数最小値":pass_number_min,
    #         "受験人数平均値":all_number_mean,
    #         "受験人数分散":all_number_std,
    #         "受験人数最大値":all_number_max,
    #         "受験人数最小値":all_number_min
    #     }

    #     feature_maker[school]=dict_tmp  
    for school in schoolname:
        df_tmp=df_test.groupby("受験校").get_group(school)
        pass_number=len(df_tmp[df_tmp["合否"]==1])
        all_number=len(df_tmp.groupby(["年度"]))
        dict_tmp={
            "合格人数":pass_number,
            "受験人数":all_number
        }

        feature_maker_test[school]=dict_tmp
    def feature_make_test_pass(x):
        return feature_maker_test[x]["合格人数"]
    def feature_make_test_all(x):
        return feature_maker_test[x]["受験人数"]

    # def feature_make_pass_number_mean(x):
    #     return feature_maker[x]["合格平均人数"]
    # def feature_make_pass_number_std(x):
    #     return feature_maker[x]["合格人数分散"]
    # def feature_make_pass_number_max(x):
    #     return feature_maker[x]["合格人数最大値"]
    # def feature_make_pass_number_min(x):
    #     return feature_maker[x]["合格人数最小値"]

    # def feature_make_all_number_mean(x):
    #     return feature_maker[x]["受験人数平均値"]
    # def feature_make_all_number_std(x):
    #     return feature_maker[x]["受験人数分散"]
    # def feature_make_all_number_max(x):
    #     return feature_maker[x]["受験人数最大値"]
    # def feature_make_all_number_min(x):
    #     return feature_maker[x]["受験人数最小値"]

    
    
    #Train

    school=schoolname
    accuracy_list=[]
    recall_list=[]
    precision_list=[]
    f1_list=[]

    accuracy_list_test=[]
    recall_list_test=[]
    precision_list_test=[]
    f1_list_test=[]
    with mlflow.start_run(run_name="训练与测试数据集 lgb"):
        train()

    # mlflow.log_param("alpha", alpha)
    # mlflow.log_param("l1_ratio", l1_ratio)
    # mlflow.log_metric("rmse", rmse)
