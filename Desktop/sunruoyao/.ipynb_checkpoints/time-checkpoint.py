
# -*- coding: utf-8 -*-
import pyscipopt as scip
import time
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE
#----------import--------------

class ShowInfo(Eventhdlr):
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
    model.readProblem("DATA/easy-sample/gen-ip002.mps")

    print("read done---")
    # start measuring time


    # set freq
    model.setIntParam("heuristics/rins/freq",-1)


    eventhdlr = ShowInfo()
    model.includeEventhdlr(eventhdlr, "ShowInfo", "PySCIPOpt Event handler to show the solving process in the output panel ")

    model.optimize()
