import pyscipopt as scip
import time
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE
#----------import--------------
class ShowInfo(Eventhdlr):
    """PySCIPOpt Event handler to show the solving process in the output panel """
    #ここでは、'includeEventhdlr' を介して出力パネルで求解プロセスの進捗状況を確認できるため、待機する求解時間を判断するのがより便利になります。

    def __init__(self):
        self.node_count = 0

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)
#-------------main-------------
if __name__ == "__main__":

    # make a SCIP model
    model = scip.Model()

    # read .mps file
    #model.readProblem("/Users/oukeikou/Desktop/sunruoyao/easy-sample/gen-ip002.mps")
    model.readProblem("/Users/oukeikou/Desktop/sunruoyao/easy-sample/markshare_4_0.mps")

    print("read done---")
    # start measuring time
    start_time = time.time()

    # set freq
    model.setIntParam("heuristics/rins/freq",3)


    eventhdlr = ShowInfo()
    model.includeEventhdlr(eventhdlr, "ShowInfo", "PySCIPOpt Event handler to show the solving process in the output panel ")

    model.optimize()

    # record the end time
    end_time = time.time()

    # calculate the processing time
    elapsed_time = end_time - start_time

    # check the solving status, and if everything is normal, output to a text file
    if model.getStatus() == "optimal":
        #optimal_solution = {var.name: model.getVal(var) for var in model.getVars()}

        node_count_fun = model.getNNodes()

        # You can change the file name here and include the total number of nodes in the output
        with open("mark_freq_3to5_1w_node.txt", "w") as f:
            f.write("Optimal Solution Found:\n")
            f.write(f"node_count_fun: {node_count_fun}\n")
            f.write(f"time: {elapsed_time}\n")
        
        # Free up space
        del model

    