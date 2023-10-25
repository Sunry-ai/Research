from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE,SCIP_HEURTIMING
import pyscipopt
import sys , os
from tqdm import tqdm
import random
import pandas as pd
import numpy as np

test = pyscipopt.scip.Model()
test.setIntParam("heuristics/rins/freq",-1)
test.readProblem("/Users/oukeikou/Desktop/sunruoyao/DATA/easy-sample/gen-ip002.mps")

print("Objective function type:", test.getObjectiveSense())
print("number of vars:", test.getNVars())

#------------------------------------------------------------------------------
class FixedVarsAtNode(Eventhdlr):
    """PySCIPOpt Event handler to write fixed vars of each node to a text file."""

    def __init__(self,random_seed):

        self.count=0
        self.count_limit=200000

        self.batch_size = 5000  # Set the number of nodes for each record in the file
        self.batch_number = 0  # The batch number for the current file

        self.data_folder = "/Users/oukeikou/Desktop/sunruoyao/DATA/ip002_output/origin"
        self.original_obj_values = self.read_data_file()

        #make random seed
        self.random_seed = random_seed
        self.random = random.Random(random_seed)

        self.nextnode = False

        #Create a new file every time 5000 data records are accumulated
        self.data_threshold = 5000

        self.cloNum = 41
        self.df_var_info = pd.DataFrame(columns=self.make_colDf(self.cloNum))

    def make_colDf(self,cloNum):
        cloName=['y_label','Node',"Depth"]
        for var_index in range(cloNum):
            var_name_prefix = str(var_index + 1)  
            var_name_prefix = var_name_prefix + "_" if var_name_prefix else ""  
            cloName += ["var_index",var_name_prefix+"type",var_name_prefix+"Lb",var_name_prefix+"Ub",
                       var_name_prefix+"Glb",var_name_prefix+"Gub"]
        return cloName

    
    def read_data_file(self):
        obj_values_numpy = np.empty((self.batch_size,))
        data = pd.DataFrame(columns=['Node', 'ObjVal'])
        file_name = os.path.join(self.data_folder, f"marks_output_{self.batch_number}_y_info.csv")
        if os.path.exists(file_name):
            data = pd.read_csv(file_name)
            obj_values_numpy = data['ObjVal'].values
            obj_values_numpy = np.insert(obj_values_numpy, 0, 0)
        return obj_values_numpy
        

    def eventexec(self, event):
        
        if self.count<self.count_limit:
            self.transvars = self.model.getVars(transformed=True)
            print(len(self.transvars))
                  
            if self.nextnode:
                #Compare for changes, Objective function type: minimize.
                obj_value_heuristic = float(self.model.getObjVal())
                self.depth = self.model.getDepth()
                
                if obj_value_heuristic < float(self.original_obj_values[self.count % self.batch_size]):
                    #Write node information
                    self.transvars = self.model.getVars(transformed=True)
                    node_info=[1,self.count,self.depth]
                    for var_index, var in enumerate(self.transvars):
                        Glb = var.getUbGlobal()#Obtain the global upper bound of the variable
                        Gub = var.getLbGlobal()
                        lb = var.getLbLocal()  
                        ub = var.getUbLocal() 
                        var_type = var.vtype()
                        node_info += [var_index,var_type,lb,ub,Glb,Gub]
                        
                        #append to df_var_info
                    node_info = pd.DataFrame([node_info],columns=self.make_colDf(self.cloNum))
                    
                    self.df_var_info = pd.concat([self.df_var_info, node_info], ignore_index=True)

                    
                #save data
                else:
                    #Write node information
                    self.transvars = self.model.getVars(transformed=True)
                    node_info=[0,self.count,self.depth]
                    for var_index, var in enumerate(self.transvars):
                        Glb = var.getUbGlobal()#Obtain the global upper bound of the variable
                        Gub = var.getLbGlobal()
                        lb = var.getLbLocal() 
                        ub = var.getUbLocal() 
                        var_type = var.vtype()
                        node_info += [var_index,var_type,lb,ub,Glb,Gub]
                        
                    #append to df_var_info
                    node_info = pd.DataFrame([node_info],columns=self.make_colDf(self.cloNum))
                   
                    self.df_var_info = pd.concat([self.df_var_info, node_info], ignore_index=True)


            else:
                pass

            use_heuristic = self.random.choice([True, False])
            if use_heuristic: #use H in the next node
                self.model.setIntParam("heuristics/rins/freq",1)
                self.nextnode = True
            else:
                self.model.setIntParam("heuristics/rins/freq",0)
                self.nextnode = False
            
            if self.count % self.data_threshold == 0:
                self.save_data()
            if self.count % self.batch_size == 0:
                # Switch to the next batch of data files
                self.batch_number += 1
                self.original_obj_values = self.read_data_file()

            self.count += 1
        else:
            pass
         
      
    def save_data(self):
        #Output as a CSV file
        file_name = 'DATA/ip002_output/withH/batch_'+str(self.batch_number) +"_info.csv"
        self.df_var_info.to_csv(file_name, index=False)
        self.df_var_info = pd.DataFrame(columns=self.make_colDf(self.cloNum))

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)
#-----------------------------------------------------------------------
if __name__ == "__main__":
    # Specify the output file path
    #output_var_info_file = "var_info.txt"
    
    test = pyscipopt.scip.Model()
    # test.setIntParam("heuristics/rins/freq",100)
    test.readProblem("/Users/oukeikou/Desktop/sunruoyao/DATA/easy-sample/gen-ip002.mps")
    print("read done")


    # Create and add event handler with the specified output file
    eventhdlr = FixedVarsAtNode(10)
    test.includeEventhdlr(eventhdlr, "FixedVarsAtNode", "Python event handler to write fixed variables after each solved node")

    # Optimize the problem
    test.optimize()
    print("optimized")
    
    # eventhdlr.write_2_file('markshare_4_0')
