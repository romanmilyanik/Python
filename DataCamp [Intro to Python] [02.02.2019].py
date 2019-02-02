a = 1.00

#find out the type of data
type(a)                                         #float

#change the data type
int(a)                                          #1
float(a)                                        #1.0
bool(a)                                         #True
str(a)                                          #'1.0'

#list
#[] first - include, last - not

#-------------------------------------------------------------------------------------
my_list = [11, 22, 33, 44, 55]

my_list[2]                                      #33
my_list[2:3]                                    #[33]
my_list[2:4]                                    #[33, 44]
my_list[:3]                                     #[11, 22, 33]
my_list[2:]                                     #[33, 44, 55]

#-------------------------------------------------------------------------------------

my_list2 = [[1, "Roman"], [2, "Zoryana"], [3, "Taras"]]       #diff. type in one list

#-------------------------------------------------------------------------------------

my_list3 = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]

my_list3[2][0]                                  #'g'
my_list3[2][:2]                                 #['g', 'h']

#-------------------------------------------------------------------------------------

my_list4 = [1, 2, 3, 4, 5]

#change list element
my_list4[2] = 22                                #[1, 2, 22, 4, 5]
#add element to list
my_list4 + [6, 7, 8]                            #[1, 2, 22, 4, 5, 6, 7, 8]
#ledete element from list
del(my_list4[1])                                #[1, 22, 4, 5]

#-------------------------------------------------------------------------------------

#create list
my_list5 = list()

#-------------------------------------------------------------------------------------

#array
import numpy as np

height = [1, 2, 3, 4, 5, 6, 7, 8]                    #list

#create np.array
np_height = np.array(height)                         #array([1, 2, 3, 4, 5, 6, 7, 8])
h = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])        #array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

h>4     #array([False, False, False, False,  True,  True,  True,  True,  True, True])
h[h>4]  #array([ 5,  6,  7,  8,  9, 10])

