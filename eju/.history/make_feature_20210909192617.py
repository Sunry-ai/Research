import pandas as pd


class Feature_maker():
    def __init__(self,path) -> None:
        self.df=pd.read_excel(path)
        
    
    def main(self):
        self.clean_raw_data(option="phyche")
        self.make_feature(option="Default")
        self.filter_school_name()
        
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
            self.df=self.df[self.df["物理"]>0]
            self.df=self.df[self.df["化学"]>0]
            #self.df=self.df[self.df["生物"]==0]

        if option=="bioche":
            self.paras=['日本語', '記述', '生物','化学', '数2',"理科总分","理科综合", 'max(全部)',"生物数学","日语数学","化学数学"]
            self.df=self.df[self.df["生物"] >0]
            self.df=self.df[self.df["化学"]> 0]
            #self.df=self.df[self.df["物理"]==0]
            
    def filter_school_name(self):
        #学校を順位付け
        info=self.df[self.df["合否"]==1].groupby("受験校").mean()[self.paras].sort_values("max(全部)")
        schoolname=list(info.index)
        schoolname.remove('千葉科学大学')
        schoolname.reverse()
        no=[]
        for i in range(len(schoolname)):
            no.append(str(i))
        id_dict=dict(zip(schoolname,no))

        self.schoolname = {int(v): k for k, v in id_dict.items()}
        
    def make_feature(self,option="Default"):
        if option=="Default":
            self.make_normal_feature()
        elif option=="stat":
            pass
    
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


