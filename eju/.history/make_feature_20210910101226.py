import pandas as pd


class Feature_maker():
    def __init__(self,path) -> None:
        self.df=pd.read_excel(path)
        
    
    def main(self):
        self.clean_raw_data(option="phyche")
        self.filter_school_name()
        self.make_features(option="Stat")
        
        
        return self.paras,self.schoolname,self.df

    def clean_raw_data(self,option="phyche"):
        #短期大学を排除
        def judge(x):
            if "短期" in x:
                return "NO"
            else:
                return x
                
        self.df.受験校=self.df.受験校.apply(judge)
    
        #学科に応じて修正
        if option=="phyche":
            self.paras=['日本語', '記述', '物理','化学', '数2',"理科总分","理科综合", 'max(全部)',"物理数学","日语数学","化学数学"]
            self.df=self.df[self.df["日本語"] >0]
            self.df=self.df[self.df["記述"] >0]
            self.df=self.df[self.df["数2"] >0]
            self.df=self.df[self.df["物理"]>0]
            self.df=self.df[self.df["化学"]>0]
            #self.df=self.df[self.df["生物"]==0]

        if option=="bioche":
            self.paras=['日本語', '記述', '生物','化学', '数2',"理科总分","理科综合", 'max(全部)',"生物数学","日语数学","化学数学"]
            
            self.df=self.df[self.df["日本語"] >0]
            self.df=self.df[self.df["記述"] >0]
            self.df=self.df[self.df["数2"] >0]
            self.df=self.df[self.df["生物"] >0]
            self.df=self.df[self.df["化学"]> 0]
            #self.df=self.df[self.df["物理"]==0]
            
    def filter_school_name(self):
        #学校を順位付け
        info=self.df[self.df["合否"]==1].groupby("受験校").mean().sort_values("max(全部)")
        schoolname=list(info.index)
        schoolname.remove('千葉科学大学')
        schoolname.reverse()
        
        no=[]
        for i in range(len(schoolname)):
            no.append(str(i))
        id_dict=dict(zip(schoolname,no))

        self.schoolname = {int(v): k for k, v in id_dict.items()}
        
        #df中去除掉合格人数为0的学校
        def school_in_list(x):
            if x in self.schoolname.values():
                return x
            else:
                return "NO"
        self.df.受験校=self.df.受験校.apply(school_in_list)
        self.df=self.df[self.df.受験校!="NO"]        
        print(self.df.受験校.unique())
    def make_features(self,option="Default"):
        if option=="Default":
            self.make_normal_feature()
        elif option=="Stat":
            feature_maker={}
            feature_maker_test={}
            for school in self.schoolname.values():
                df_tmp=self.df.groupby("受験校").get_group(school)

                pass_number_mean=df_tmp[df_tmp["合否"]==1].groupby(["年度"]).count().mean()["合否"]
                pass_number_std=df_tmp[df_tmp["合否"]==1].groupby(["年度"]).count().std()["合否"]
                pass_number_max=df_tmp[df_tmp["合否"]==1].groupby(["年度"]).count().max()["合否"]
                pass_number_min=df_tmp[df_tmp["合否"]==1].groupby(["年度"]).count().min()["合否"]
                
                all_number_mean=df_tmp.groupby(["年度"]).count().mean()["合否"]
                all_number_std=df_tmp.groupby(["年度"]).count().std()["合否"]
                all_number_max=df_tmp.groupby(["年度"]).count().max()["合否"]
                all_number_min=df_tmp.groupby(["年度"]).count().min()["合否"]
            
                dict_tmp={
                    "合格平均人数":pass_number_mean,
                    "合格人数分散":pass_number_std,
                    "合格人数最大値":pass_number_max,
                    "合格人数最小値":pass_number_min,
                    "受験人数平均値":all_number_mean,
                    "受験人数分散":all_number_std,
                    "受験人数最大値":all_number_max,
                    "受験人数最小値":all_number_min
                }

                feature_maker[school]=dict_tmp  
            for school in self.schoolname.values():
                df_tmp=self.df.groupby("受験校").get_group(school)
                pass_number=len(df_tmp[df_tmp["合否"]==1])
                all_number=len(df_tmp.groupby(["年度"]))
                dict_tmp={
                    "合格人数":pass_number,
                    "受験人数":all_number
                }

                feature_maker_test[school]=dict_tmp
            print(feature_maker.keys())
            def feature_make_pass_number_mean(x):
                return feature_maker[x]["合格平均人数"]
            def feature_make_pass_number_std(x):
                return feature_maker[x]["合格人数分散"]
            def feature_make_pass_number_max(x):
                return feature_maker[x]["合格人数最大値"]
            def feature_make_pass_number_min(x):
                return feature_maker[x]["合格人数最小値"]
            def feature_make_all_number_mean(x):
                return feature_maker[x]["受験人数平均値"]
            def feature_make_all_number_std(x):
                return feature_maker[x]["受験人数分散"]
            def feature_make_all_number_max(x):
                return feature_maker[x]["受験人数最大値"]
            def feature_make_all_number_min(x):
                return feature_maker[x]["受験人数最小値"]
            self.df["合格平均人数"]=self.df["受験校"].apply(feature_make_pass_number_mean)
            self.df["合格人数分散"]=self.df["受験校"].apply(feature_make_pass_number_std)
            self.df["合格人数最大値"]=self.df["受験校"].apply(feature_make_pass_number_max)
            self.df["合格人数最小値"]=self.df["受験校"].apply(feature_make_pass_number_min)
            self.df["受験人数平均値"]=self.df["受験校"].apply(feature_make_all_number_mean)
            self.df["受験人数分散"]=self.df["受験校"].apply(feature_make_all_number_std)
            self.df["受験人数最大値"]=self.df["受験校"].apply(feature_make_all_number_max)
            self.df["受験人数最小値"]=self.df["受験校"].apply(feature_make_all_number_min)
            self.paras.append("合格平均人数")
            self.paras.append("合格人数分散")
            self.paras.append("合格人数最大値")
            self.paras.append("合格人数最小値")
            self.paras.append("受験人数平均値")
            self.paras.append("受験人数分散")
            self.paras.append("受験人数最大値")
            self.paras.append("受験人数最小値")

    def make_normal_feature(self,option="phyche"):
        if option=="phyche":
            self.df["理科综合"]=self.df['物理']+self.df['化学']
            self.df["理科总分"]=self.df['物理']+self.df['化学']+self.df['数2']
            self.df["物理数学"]=self.df['物理']+self.df['数2']
            self.df["日语数学"]=self.df['日本語']+self.df['数2']
            self.df["化学数学"]=self.df['化学']+self.df['数2']
        elif option=="bioche":
            self.df["理科综合"]=self.df['化学']+self.df['生物']
            self.df["理科总分"]=self.df['化学']+self.df['生物']+self.df['数2']
            self.df["生物数学"]=self.df['生物']+self.df['数2']
            self.df["日语数学"]=self.df['日本語']+self.df['数2']
            self.df["化学数学"]=self.df['化学']+self.df['数2']


