import pyscipopt as scip
import random
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE

rins_heuristic = pyscipopt.heuristics.Rins()


class FixedVarsAtNode(Eventhdlr):
    """PySCIPOpt Event handler to write fixed vars of each node to a text file."""

    def __init__(self,random_seed):
        
        #define the path of file
        # self.output_var_info_file = "var_info/var_info.txt"
        
        #others
        self.df_y_info = pd.DataFrame(columns=['Before', 'After', 'y_label'])

        self.count=0
        self.count_limit=5
        
        #make random seed
        self.random_seed = random_seed
        self.random = random.Random(random_seed)

    def eventexec(self, event):
        if self.count<self.count_limit:
            
            # 创建一个空的DataFrame
            df_var_info = pd.DataFrame(columns=['Var', 'Type', 'LowerBound','UpperBound','GLB','GUB'])
            
            
            fixedvars=[]
            # get node number
            nodenumber = event.getNode().getNumber()
        
            #use_heuristic
            use_heuristic = self.random.choice([True, False])
            if use_heuristic:
                #     # 保存未使用启发式时的目标函数值
                # obj_value_no_heuristic = self.model.getObjVal() 
                #写入node信息
                self.transvars = self.model.getVars(transformed=True)
                for var in self.transvars:
                    var_name = var.name
                    Glb = var.getUbGlobal()#获取变量的全局上界
                    Gub = var.getLbGlobal()#获取变量的全局下界
                    lb = var.getLbLocal()  # 获取变量的局部下界
                    ub = var.getUbLocal()  # 获取变量的局部上界
                    var_type = var.vtype()
                    
                    var_info_add_df ={'Var': var_name, 'Type': var_type, 'LowerBound': lb, 'UpperBound': ub, 'GLB':Glb, 'GUB':Gub}
                    #append to df_var_info
                    df_var_info = df_var_info.append(var_info_add_df, ignore_index=True)
                
                #输出为csv文件
                file_name='var_info/'+str(self.count) +"var_info.csv"
                df_var_info.to_csv(file_name, index=False) 
                
               
                # obj_value_heuristic = self.model.getObjVal() 
                # if_y_better = self.check_if_better(obj_value_no_heuristic,obj_value_heuristic)
                #每个node运行后就直接存入对应的行，最后全运完直接把文件导出
                y_info_add_df ={'Before': obj_value_no_heuristic, 'After': obj_value_heuristic, 'y_label': if_y_better}
                self.df_y_info.append(y_info_add_df, ignore_index=True)
            else: 
                pass
                
            self.count+=1
        else:
             pass
       
    def check_if_better(self,before,after):
        obj_fun=self.model.getObjectiveSense()
        
        if obj_fun == 'minimize':
            if before>after:
                if_better = 1
            else:
                if_better = 0
        else:
            if before < after:
                if_better = 1
            else:
                if_better = 0
        return if_better
                    
    def write_y_info_file(self , info):
        #输出为csv文件
        file_name = str(self.count) +"_y_info.csv"
        self.df_y_info.to_csv(file_name, index=False)
        
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)
