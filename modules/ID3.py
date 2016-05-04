import math
from node import Node
from math import log
import operator
import sys

def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    '''
    See Textbook for algorithm.
    Make sure to handle unknown values, some suggested approaches were
    given in lecture.
    ========================================================================================================
    Input:  A data_set, attribute_metadata, maximum number of splits to consider for numerical attributes,
	maximum depth to search to (depth = 0 indicates that this node should output a label)
    ========================================================================================================
    Output: The node representing the decision tree learned over the given data set
    ========================================================================================================

    '''
    # Your code here
    pass

def check_homogenous(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Checks if the output value (index 0) is the same for all examples in the the data_set, if so return that output value, otherwise return None.
    ========================================================================================================
    Output: Return either the homogenous attribute or None
    ========================================================================================================
     '''
    # Your code here
    for i in xrange(len(data_set)-1):
        if not data_set[0][0] == data_set[i+1][0]:
            return None
    return data_set[0][0]

# ======== Test Cases =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  None
# data_set = [[0],[1],[None],[0]]
# check_homogenous(data_set) ==  None
# data_set = [[1],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  1

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, splits counts for numeric
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero
    ========================================================================================================
    Output: best attribute, split value if numeric
    ========================================================================================================
    '''
    gain_ratio = 0.0
    max_gain_ratio = 0.0
    best_attribute = 0
    split_value = 0.0
    final_split_value = 0.0
    for i in xrange(1,len(attribute_metadata)):
        if attribute_metadata[i].values()[0]:      # the attribute is nominal
            gain_ratio = gain_ratio_nominal(data_set,i)
        else:       
            if numerical_splits_count[i]>0:        # the attribute is numeric
                gain_ratio_result = gain_ratio_numeric(data_set,i,1)
                gain_ratio = gain_ratio_result[0]
                split_value = gain_ratio_result[1]
            else:
                gain_ratio = 0.0
        if gain_ratio > max_gain_ratio:
                best_attribute = i
                max_gain_ratio = gain_ratio
                final_split_value = split_value
    if max_gain_ratio == 0:
        return (False,False)
    elif attribute_metadata[best_attribute].values()[0]:
        return (best_attribute,False)
    else:
        return (best_attribute,final_split_value)



# # ======== Test Cases =============================
# numerical_splits_count = [20,20]
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
# data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
# data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)

# Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.

def mode(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Takes a data_set and finds mode of index 0.
    ========================================================================================================
    Output: mode of index 0.
    ========================================================================================================
    '''
    # Your code here
    total = 0
    for i in xrange(len(data_set)):
        total += data_set[i][0];
    if(total > len(data_set)/2):
        return 1
    else:
        return 0
    
# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# mode(data_set) == 1
# data_set = [[0],[1],[0],[0]]
# mode(data_set) == 0

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. See Textbook for formula
    ========================================================================================================
    '''
    
    length = len(data_set)
    total = 0
    for i in xrange(len(data_set)):
        total += data_set[i][0];
    p1 = float(total) / float(length) 
    p0 = 1 - p1
    if p1==0 or p0==0:
        return 0
    return -(math.log(p1,2)*p1 + math.log(p0,2)*p0)
    

# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
# entropy(data_set) == 0.811
# data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
# entropy(data_set) == 1.0
# data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
# entropy(data_set) == 0


def gain_ratio_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, index for a nominal attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    length = len(data_set)
    H_Ex = entropy(data_set)   # H(Ex) = entropy for data_set

    dic = split_on_nominal(data_set,attribute)
    value_set = []     # times of occurence for each value of a given attribute
    occur = 0          # total times of occurence for a given attribute
    IV = 0.0
    H_total = 0.0
    for i in xrange(len(dic)):
        value_set.append(len(dic.values()[i]))
        occur += value_set[i]
    for i in xrange(len(value_set)):
        p = float(value_set[i])/float(occur)
        IV -= p * my_log(p,2)   # intrinsic value
    for i in xrange(len(dic)):
        count = len(dic.values()[i])
        countOf1 = 0
        for j in xrange(count):
            countOf1 += dic.values()[i][j][0]
        p1 = float(countOf1)/float(count)
        p0 = 1 - p1
        H_total -= float(count)/float(length)*(p1 * my_log(p1,2) + p0 * my_log(p0,2)) 
    return (H_Ex - H_total)/IV

# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151
# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741
# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.
    ========================================================================================================
    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
            Use the equation here: https://en.wikipedia.org/wiki/Information_gain_ratio
            And restrict your search for possible thresholds to examples with array index mod(step) == 0
    ========================================================================================================
    Output: This function returns the gain ratio and threshold value
    ========================================================================================================
    '''
    i = 0
    h_ex = entropy(data_set)  #entropy
    while i < len(data_set):
        temp_thre = data_set[i][attribute]
        temp_tuple = split_on_numerical(data_set, attribute, temp_thre)
        #iv calculation
        small_p = float(len(temp_tuple[0]))/float(len(data_set))
        large_p = 1 - small_p
        iv = -(my_log(small_p,2)*small_p + my_log(large_p,2)*large_p) #TODO handle p==0 case
        small_posi = 0
        large_posi = 0
        for j in xrange(len(temp_tuple[0])):
            if temp_tuple[0][j][0] == 1:
                small_posi += 1
        if len(temp_tuple[0]) == 0:
            p0 = 0
        else:
            p0 = float(small_posi)/float(len(temp_tuple[0]))
        p1 = 1- p0
        for j in xrange(len(temp_tuple[1])):
            if temp_tuple[1][j][0] == 1:
                large_posi += 1
        if len(temp_tuple[1]) == 0:
            p0 = 0
        else:
            p2 = float(large_posi)/float(len(temp_tuple[1]))
        p3 = 1 - p2
        ha_ex = -small_p*(p0*my_log(p0,2)+p1*my_log(p1,2)) - large_p*(p2*my_log(p2,2)+p3*my_log(p3,2))
        if iv == 0:
            temp_gain = 0
        else:
            temp_gain = float(h_ex-ha_ex)/iv
        if i == 0:
            gain = {temp_thre:temp_gain}
        else:
            gain.update({temp_thre:temp_gain})
        i += steps
    sorted_x = sorted(gain.items(), key=operator.itemgetter(1))
    # print sorted_x
    threshold = sorted_x[len(sorted_x)-1][0]
    gain_ratio = sorted_x[len(sorted_x)-1][1]
    return (gain_ratio, threshold)
# ======== Test case =============================
# data_set,attr,step = [[0,0.05], [1,0.17], [1,0.64], [0,0.38], [0,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
# gain_ratio_numeric(data_set,attr,step) == (0.31918053332474033, 0.64)
# data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
# gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)
# data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
# gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)

def split_on_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  subset of data set, the index for a nominal attribute.
    ========================================================================================================
    Job:    Creates a dictionary of all values of the attribute.
    ========================================================================================================
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    ========================================================================================================
    '''
    length = len(data_set)
    for i in xrange(length):
      #  index0_set[i].append(data_set[i][0])
        if i == 0:
            dic = {data_set[i][attribute]:[data_set[i]]}
        elif dic.has_key(data_set[i][attribute]):
            dic[data_set[i][attribute]] += [data_set[i]]
        else:
            dic_add = {data_set[i][attribute]:[data_set[i]]}
            dic.update(dic_add)
    return dic
# ======== Test case =============================
# data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
# split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]}
# data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
# split on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]}

def split_on_numerical(data_set, attribute, splitting_value):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, threshold (splitting) value
    ========================================================================================================
    Job:    Splits data_set into a tuple of two lists, the first list contains the examples where the given
	attribute has value less than the splitting value, the second list contains the other examples
    ========================================================================================================
    Output: Tuple of two lists as described above
    ========================================================================================================
    '''
    small = list()
    large = list()
    for i in xrange(len(data_set)):
        if data_set[i][attribute] < splitting_value:
            small.append(data_set[i])
        else:
            large.append(data_set[i])
    v = small, large
    return v
# ======== Test case =============================
# d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
# split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]])
# d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
# split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]])

def my_log(d, b):
    if d == 0:
        return 0
    else:
        return math.log(d, b)

