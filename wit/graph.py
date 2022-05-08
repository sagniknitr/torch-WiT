from importlib.util import LazyLoader
import graphviz


class Task(object):
    def __init__(self, node_id, node_desc="", forwared_pass_tim=0.0, backward_pass_time=0.0):
        self.node_id = node_id
        self.node_desc = node_desc
        self.depth = None
        self.height = None


class Task(object):
    def __init__(self, execution_thread, duration=0.0, gap=0.0, layer=""):
        self.execution_thread = execution_thread
        self.layer = layer
        self.duration = duration
        self.gap = gap

class CPUTask(Task):
    pass

class DataLoadTask(Task):
    pass

class CommunicationTask(Task):
    pass

import json
import networkx as nx
import matplotlib.pyplot as plt

class graphNode:
    proc_type = 'cpu'
    proc_ph = 'X'
    proc_name = 'name'
    proc_pid = 0
    proc_tid = 0
    proc_ts = 0
    proc_dur = 0
    
    def _print_node(self):
        print(' Type: ', self.proc_type, ' Ph: ', self.proc_ph, ' Name: ', self.proc_name, ' PID: ', self.proc_pid,
        ' TID: ', self.proc_tid, ' Timestamp: ', self.proc_ts, ' Duration: ', self.proc_dur)
    
    def construct_node(self, proc):
        self.proc_ph = proc['ph']
        self.proc_type = proc['cat']
        self.proc_ph = proc['name']
        self.proc_pid = proc['pid']
        self.proc_tid = proc['tid']
        self.proc_ts = proc['ts']
        self.proc_dur = proc['dur']
        

def printNodes(_nodeList):
    for node in _nodeList:
        node._print_node()


f = open('trace.json')
data = json.load(f)
nodeList = []
dep_graph = nx.Graph()
 
for proc in data['traceEvents']:
    if 'cat' in proc:
        new_node = graphNode()
        new_node.construct_node(proc)
        nodeList.append(new_node)
        dep_graph.add_node(new_node)

f.close()

nodeList.sort(key = lambda proc: proc.proc_ts)
printNodes(nodeList)

for nd_1 in nodeList:
    for nd_2 in nodeList:
        if (nd_1.proc_ts + nd_1.proc_dur >= nd_2.proc_ts) and (nd_2.proc_ts <= nd_1.proc_ts):
            dep_graph.add_edge(nd_1, nd_2)

nx.draw(dep_graph)
plt.show()