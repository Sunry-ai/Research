import pyscipopt as scip
import time
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE

#----------import------------------

class TestFreqChange(Eventhdlr):
    """PySCIPOpt Event handler to write fixed vars of each node to a text file."""

    def __init__(self):
        
        self.node_count = 0
        #define the path of output file
        self.file="mark2_freq_3to5_1w_node.txt"
        
    def eventexec(self,event):
        # Change the freq of Rins after 1000 nodes
        if self.node_count <= 1000:
            self.model.setIntParam("heuristics/rins/freq",3)

        else:
            # change the freq to 5
            self.model.setIntParam("heuristics/rins/freq",5)

            #The following two lines of code are for testing whether the if branch has been entered correctly - confirming entry.
            # with open(self.file, "a") as f:
            #     f.write(f"node: {self.node_count}\n")
            
        self.node_count += 1

    def write(self):
        # You can change the file name here and include the total number of nodes in the output
        with open("mark_freq_3to5_1w_node.txt", "w") as f:
            f.write(f"Objective value: {self.model.getObjVal()}\n")
            f.write(f"node_count_fun: {self.node_count}\n")

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)

#---------------Class-------------------------

if __name__ == "__main__":

    # make a SCIP model
    model = scip.Model()

    # read .mps file
    #model.readProblem("/Users/oukeikou/Desktop/sunruoyao/easy-sample/gen-ip002.mps")
    model.readProblem("/Users/oukeikou/Desktop/sunruoyao/easy-sample/markshare_4_0.mps")

    print("read done---")
    start_time = time.time()

    model.setIntParam("heuristics/rins/freq",3)


    eventhdlr = TestFreqChange()
    model.includeEventhdlr(eventhdlr, "FixedVarsAtNode", "Python event handler to write fixed variables after each solved node")

    model.optimize()

    eventhdlr.write()

    # start measuring time
    end_time = time.time()

    # calculate the processing time
    elapsed_time = end_time - start_time

    if model.getStatus() == "optimal":

        eventhdlr.write()
        # Free up space
        del model