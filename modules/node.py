# DOCUMENTATION
# =====================================
# Class node attributes:
# ----------------------------
# children - a list of 2 nodes if numeric, and a dictionary (key=attribute value, value=node) if nominal.  
#            For numeric, the 0 index holds examples < the splitting_value, the 
#            index 1 holds examples >= the splitting value
#
# label - is None if there is a decision attribute, and is the output label (0 or 1 for
#	the homework data set) if there are no other attributes
#       to split on or the data is homogenous
#
# decision_attribute - the index of the decision attribute being split on
#
# is_nominal - is the decision attribute nominal
#
# value - Ignore (not used, output class if any goes in label)
#
# splitting_value - if numeric, where to split
#
# name - name of the attribute being split on

class Node:
    def __init__(self):
        # initialize all attributes
        self.label = None
        self.decision_attribute = None
        self.is_nominal = None
        self.value = None
        self.splitting_value = None
        self.children = {}
        self.name = None

    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''
        current_node = self
        while current_node.label == None:
            if current_node.is_nominal == False:     # current node is numeric
                if instance[current_node.decision_attribute] < current_node.splitting_value:
                    current_node = current_node.children[0]
                else:
                    current_node = current_node.children[1]
            else:
                for key in current_node.children:    # current node is nominal
                    if key == instance[current_node.decision_attribute]:
                        current_node = current_node.children.get(key)
                        break
        return current_node.label

    def print_tree(self, indent = 0):
        '''
        returns a string of the entire tree in human readable form
        IMPLEMENTING THIS FUNCTION IS OPTIONAL
        '''
        # Your code here
        pass

#still need to test more corner cases if needed
    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''
        current = self
        s = []
        s.append(current)
        s.append(str(current.name))

        while len(s) != 0:
            path = s.pop()
            current = s.pop()

            if current.label == 1 or current.label == 0:  #------------------delete?
                if current.label == 1:
                    print ' %s ' % path

            if current.is_nominal == False:
                if current.children[1]:
                    rightstr = path + ">= "+ "%s" % current.splitting_value
                    if current.children[1].name != None:
                        rightstr += "^" + str(current.children[1].name)
                    s.append(current.children[1])
                    s.append(rightstr)

                if current.children[0]:
                    leftstr =  path + "<"+ "%s" % current.splitting_value
                    if current.children[0].name != None:
                        leftstr += "^" + str(current.children[0].name)
                    s.append(current.children[0])
                    s.append(leftstr)
            else:
                for key in current.children:
                    str_nomi = path + "="+ "%s" % key
                    if current.children.get(key).name != None:
                        str_nomi +=  "^" + str(current.children.get(key).name)
                    s.append(current.children.get(key))
                    s.append(str_nomi)
# used to test dnf print
def check_dnf():
    a0 = Node()
    b0 = Node()
    b1 = Node()
    # b2 = Node()
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
    a0.print_dnf_tree()

check_dnf()
