
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE,SCIP_HEURTIMING
import pyscipopt
import sys
from tqdm import tqdm
import random
import pandas as pd

class FixedVarsAtNode(Eventhdlr):
    """PySCIPOpt Event handler to write fixed vars of each node to a text file."""

    def __init__(self):
        
        #define the path of file
        # self.output_var_info_file = "var_info/var_info.txt"
        
        #others
        self.orgin_sol_info = pd.DataFrame(columns=['Node', 'ObjVal'])

        self.count=0
        self.count_limit=200000

        self.batch_size = 5000  # Set the number of nodes for each record in the file
        self.batch_number = 0  # The batch number for the current file
        

    def eventexec(self, event):
        if self.count < self.count_limit:
            
            obj_value_no_heuristic = self.model.getObjVal() 
            info_add_df ={'Node': self.count, 'ObjVal': obj_value_no_heuristic}
            self.orgin_sol_info = self.orgin_sol_info.append(info_add_df,ignore_index=True)

            # Determine if the record's node count has been reached
            if self.count % self.batch_size == 0:
                # write to file
                self.write_2_file(f'DATA/ip054_output/origin/ip054_output_{self.batch_number}')
                self.batch_number += 1
                # make a new DataFrame
                self.orgin_sol_info = pd.DataFrame(columns=['Node', 'ObjVal'])

            self.count += 1
        else:
            pass

         
      
    def write_2_file(self , filename):
        #Output as a CSV file
        file_name = filename +"_y_info.csv"
        self.orgin_sol_info.to_csv(file_name, index=False)
        

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)


if __name__ == "__main__":
    # Specify the output file path
    #output_var_info_file = "var_info.txt"
    
    test = pyscipopt.scip.Model()
    test.setIntParam("heuristics/rins/freq",0)
    test.readProblem("/Users/oukeikou/Desktop/sunruoyao/DATA/easy-sample/gen-ip054.mps")
    print("read done")


    # Create and add event handler with the specified output file
    eventhdlr = FixedVarsAtNode()
    test.includeEventhdlr(eventhdlr, "FixedVarsAtNode", "Python event handler to write fixed variables after each solved node")

    # Optimize the problem
    test.optimize()
    print("optimized")
    
    # eventhdlr.write_2_file('markshare_4_0')