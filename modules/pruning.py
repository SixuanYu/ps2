from node import Node
from ID3 import *
from operator import xor

# Note, these functions are provided for your reference.  You will not be graded on their behavior,
# so you can implement them as you choose or not implement them at all if you want to use a different
# architecture for pruning.

def reduced_error_pruning(root,training_set,validation_set,attribute_metadata):
    '''
    take the a node, training set, and validation set and returns the improved node.
    You can implement this as you choose, but the goal is to remove some nodes such that doing so improves validation accuracy.
    NOTE you will probably not need to use the training set for your pruning strategy, but it's passed as an argument in the starter code just in case.
    '''
    threshold = 0.001
    pre_acc = validation_accuracy(root,validation_set,attribute_metadata)

    while True:
        nodes = []            # a collection of all nodes
        leaves = []           # a collection of all leaves
        children_leaf_set = []     # contains all children leaves of a specific node
        collect_node_leaf(root, nodes, leaves)     
        #max_acc = pre_acc
        prune_this = None     # contains the newest node that can be pruned 
        cannot_prune = 0      # the number of nodes that cannot be pruned

        for i in range(len(nodes)):     # traverse from the last node to first node 
            nodes[len(nodes)-i-1].convert_to_leaf()
            post_acc = validation_accuracy(root,validation_set,attribute_metadata)
            if pre_acc-post_acc < threshold:
                prune_this = nodes[len(nodes)-i-1]
            else: 
                cannot_prune += 1
            nodes[len(nodes)-i-1].convert_back_to_node()
        if cannot_prune == len(nodes):    # none of nodes can be pruned
            return root
        else:
            prune_this.convert_to_final_leaf()
            collect_node_leaf(prune_this,[],children_leaf_set)
            prune_this.label = mode(children_leaf_set)


def validation_accuracy(tree, validation_set, attribute_metadata):
    '''
    takes a tree and a validation set and returns the accuracy of the set on the given tree
    '''
    validation_set = data_prep(validation_set, attribute_metadata)
    length = len(validation_set)
    if length == 0:
        return 0
    for i in xrange(length):
        if tree.classify(validation_set[i]) == validation_set[i][0]:
            count += 1
    return (count/float(length))

def collect_node_leaf(root, nodes, leaves):
    '''
    put nodes together; put leaves together
    '''
    if not root.label == None:
        leaves.append(root)
    else:
        nodes.append(root)
        for child in root.children:
            if type(root.children) == list:
                collect_node_leaf(child, nodes, leaves)
            else:
                collect_node_leaf(root.children[child], nodes, leaves)


a0 = Node()
b0 = Node()
b1 = Node()
c0 = Node()
c1 = Node()
c2 = Node()
c3 = Node()
c0.label = 0
c1.label = 1
c2.label = 0
c3.label = 1
a0.name = "weather"
a0.is_nominal = True
a0.label = None
b0.name = "#injury"
b0.label = None
b0.is_nominal = False
b0.splitting_value = 50
b0.children = [c0, c1]
b1.name = "#audience"
b1.label = None
b1.is_nominal = False
b1.splitting_value = 20.9
b1.children = [c0, c1]
a0.children = {1: b0, -1: b1}

nodes = []
leaves = []
collect_node_leaf(a0,nodes,leaves)
#for i in range(len(nodes)):
#    print nodes[i].name
for i in range(5):
    print 5-i-1