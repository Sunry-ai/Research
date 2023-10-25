from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE,SCIP_HEURTIMING
import pyscipopt
import sys , os
import time
import random
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

class FixedVarsAtNode(Eventhdlr):
    """PySCIPOpt Event handler to write fixed vars of each node to a text file."""

    def __init__(self,window):

        self.count=0
        self.count_limit=-1

        self.MLmodel_path = "model/ip054_model/ip054_lgb_10k_window_Regre.pkl"
        
        self.MLmodel = joblib.load(self.MLmodel_path)
        print(type(self.MLmodel))

        self.nextnode = False
        self.window = window

        self.cloNum = 30
        self.data = pd.DataFrame(columns=self.make_colDf(self.cloNum))

        self.prob = 0

    def make_colDf(self,cloNum):
        cloName=['Node',"Depth"]
        for var_index in range(cloNum):
            var_name_prefix = str(var_index + 1)  # +1是因为索引从0开始
            var_name_prefix = var_name_prefix + "_" if var_name_prefix else ""  # 添加下划线，如果有前缀
            cloName += ["var_index",var_name_prefix+"type",var_name_prefix+"Lb",var_name_prefix+"Ub",
                       var_name_prefix+"Glb",var_name_prefix+"Gub"]
        return cloName

    def eventexec(self, event):
        if self.count > self.count_limit:
            
            random_seed = random.random()
            if random_seed < self.prob :
                self.model.setIntParam("heuristics/rins/freq",0)
            else:
                self.model.setIntParam("heuristics/rins/freq",-1)
            
            if self.count % self.window == 1 :#收集数据
                #start_time=time.time()
                
                self.transvars = self.model.getVars(transformed=True)
                self.depth = self.model.getDepth()
                       
                node_info=[self.count,self.depth]
                for var_index, var in enumerate(self.transvars):
                    Glb = var.getUbGlobal()  # 获取变量的全局上界
                    Gub = var.getLbGlobal()  # 获取变量的全局下界
                    lb = var.getLbLocal()    # 获取变量的局部下界
                    ub = var.getUbLocal()    # 获取变量的局部上界
                    var_type = 0 if var.vtype() == 'BINARY' else 1
                
                    node_info += [var_index,var_type,lb,ub,Glb,Gub]
                self.data = pd.DataFrame([node_info],columns=self.make_colDf(self.cloNum))
                #end_time=time.time()
                #print("储存1号数据的时间:", end_time-start_time)
                
            if (self.count!=0) & (self.count % self.window == 0) :#收集数据+预测
                #start_time=time.time()
                
                self.transvars = self.model.getVars(transformed=True)
                self.depth = self.model.getDepth()
                       
                node_info=[self.count,self.depth]
                for var_index, var in enumerate(self.transvars):
                    Glb = var.getUbGlobal()  # 获取变量的全局上界
                    Gub = var.getLbGlobal()  # 获取变量的全局下界
                    lb = var.getLbLocal()    # 获取变量的局部下界
                    ub = var.getUbLocal()    # 获取变量的局部上界
                    var_type = 0 if var.vtype() == 'BINARY' else 1
                
                    node_info += [var_index,var_type,lb,ub,Glb,Gub]
                    
                node_info = pd.DataFrame([node_info],columns=self.make_colDf(self.cloNum))
                self.data = pd.concat([self.data, node_info], ignore_index=True)
                #end_time=time.time()
                #print("储存0号数据的时间:", end_time-start_time)
                #print("count:", self.count)

                #预测预测
                diff_df = self.data.diff().iloc[1:]
                
                y_pred_prob = self.MLmodel.predict(diff_df, num_iteration=30)
                #这个概率是未来window个数点内出现启发式更好解的概率 y=1的个数/window
                self.prob = y_pred_prob[0]

            self.count += 1
        else:
            pass  

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)

if __name__ == "__main__":
    
    test = pyscipopt.scip.Model()
    test.readProblem("DATA/easy-sample/gen-ip054.mps")
    print("read done")

    # Create and add event handler with the specified output file
    eventhdlr = FixedVarsAtNode(window=10000)
    test.includeEventhdlr(eventhdlr, "FixedVarsAtNode", "Python event handler to write fixed variables after each solved node")
