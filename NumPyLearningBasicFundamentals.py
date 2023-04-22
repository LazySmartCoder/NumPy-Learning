'''
THIS NUMPY LEARNING SCRIPT IS PURELY MADE BY ANIRBAN BHATTACHARYA AS NOTES.
IT IS REQUESTED TO THE VIEWER NOT TO TRY READING ANY OF IT BECAUSE IT CAN ONLY BE
UNDERSTOOD BY ANIRBAN BHATTACHARYA. THANK YOU.

AUTHOR: ANIRBAN BHATTACHARYA
GITHUB USERNAME: @LazySmartCoder
'''

import numpy as np
# Creating an array in numpy
arr = np.array([1, 4 ,8, 10])
# Using bits in numpy by giving a new arguement
arr = np.array([1, 4, 128, 10], dtype=np.int64)
# for example int8 can store values between -128 and 127 and int	int16 Integer (-32768 to 32767)

# how to locate a particular element in an array
narr = np.array([[1, 4 ,8, 10]])
narr[0, 3]

# how to find the structure of an array
narr = np.array([[1, 4 ,8, 10, 100]])
narr.shape

# how to get the dtype of an array (default dtype of an array is int 64 bit only)
narr = np.array([[1, 4 ,8, 10, 100]])
narr.dtype

#how to change a particular element
narr = np.array([[1, 4 ,8, 10, 100]])
narr[0, 2] = 96

# few funtions to know about the information of an array
marr = np.array([[1, 4 ,8, 10, 100], [75, 84, 984, 9, 10], [84, 888, 843, 2, 96]])
marr.shape
marr.size
marr.dtype

# creating arrays in numpy
zeros = np.zeros((2, 6)) # this function fills the array with zeros and it takes only one arguement i.e the shape only in the this format (())
np.zeros((2, 2, 2, 2, 2)) # here except the last 2 all the tuple items will form an array in array which is nested array. and the last item of tuple is the items in that array
rng = np.arange(20) # this function fills the array with the range till the parameter is given. it takes one arguement i.e the range of the loop
ls = np.linspace(3, 90, 8) # this gives the number of values (3rd parameter) between first parameter and second parameter with equal difference
empty = np.empty((4, 3)) # this creates a randome array with random numbers to operate some dummy operations
emptyLike = np.empty_like(rng) # this gives an array of same size of random numbers like the array as given
# note: we do all such things for efficieny told by harry bhai
ide = np.identity(10) # this gives an array with matrics as arguement cross arguement i.e if arguement is 3 so the array will be 3d and columns will also be 3
arr = np.arange(99)
arr.reshape(9, 11) # this function don't updates the array but reshapes it into the multiples. the first arguements takes the number of items in list and second parameter takes the number of arrays to get distributed
arr.ravel() # this don't updates the array but resembles the array in a straight format into a 1d array


'''
  What is an axis in numpy??
  In a cartesian plane X and Y axis are there. In the same way numpy has the concept of axis which
  represents columns and rows in an array. There are two types of axis in numpy like X and Y in
  statistics and maths i.e axis=0 and axis=1. Axis 0 is always a columns and axis 1 is always a row.
  columns are standing and rows are sleeping. Axis system in numpy is similar to rubiks cube. For more
  details regarding numpy axis system do watch CodeWithHarry numpy tutorial once again that axis concept.
  Video link - https://www.youtube.com/watch?v=Rbh1rieb3zc
'''
# some functions of numpy regading the axis concept
l = [[1, 2 ,3], [4, 8, 9], [7, 4, 9], [2, 5, 0]]
ar = np.array(l)
ar.sum(axis=0) # this adds the columns
ar.sum(axis=1) # this adds the rows
ar.T # this functions transposes the array. Means it converts the columns to row and vice versa.
ar.flat # this function prints all the values in a row format. when executed using a loop.
ar.ndim # shows the total number of dimenssions
ar.size # shows the number of items in array
ar.nbytes # shows the total bits consumed by a particular array
ar.argmax() # this function shows the maximum value's index
ar.argmin() # this function shows the minimum value's index
ar.argsort() # this sorts the list in ascending order and returns a an index array accordingly
'''We can also using argmin max and sort operation in 2d arrays.'''
a = np.array([[1, 2, 9], [7, 8, 5]])
a.argsort(axis=0)

arr1 = np.array([[1, 2, 3], [4, 9, 7], [8, 9, 0]])
arr2 = np.array([[8, 9, 0], [7, 10, 3], [4, 6, 2]])
arr1 * arr2
'''We can perform some simple mathematical operators like addition, subtraction, multiplication, division and etc'''
np.sqrt(arr1) # this function of numpy gives the square root of all integral values in a array and returns an array having all the square roots.
arr1.max() # this function directly returns the value of the maximum value not the indice.
np.where(arr1%2 == 0) # this function shows the position of a value as per the condition. The position is shows as [1, 2, 3] [5, 6, 2] so the first one is column and second one is items index. For example [0][2] means the column is 0 and item number is 2
np.count_nonzero(arr1) # this function just shows you the count of all the non zero items in an array
np.nonzero(arr1) # this function shows the indexing of items which are not zero in an array

'''Changing an array item'''
arr1[1, 1] = 900
import sys
'''Memory management skills of numpy'''
a1 = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
a2 = np.array(a1)
a2.itemsize * a2.size, a2.nbytes
sys.getsizeof(a1) * len(a1)
# nbytes is similar to itemsize * size
# here in numpy array getsizeof is not used because it returns the size of non elements items also. but we have to use in list because there is not other option.
# getsizeof just shows the size of each element not the size of list. so we have to multiply each element's size with number of items to get the actual size of full list

arr = np.eye(5, 5, k = 1, dtype = int) # This function creates a diagram of 1. of specified rows and columns and that K parameter takes the index from where to start default is 0

'''Creating some random values in more precise manner in numpy.'''
arr = np.random.rand(5, 5, 2) # This functions generates random numbers from 0 to 1.
arr = np.random.randn(5, 5, 2) # this function generates number which are close to 0 it can be positive or negative as well
arr = np.random.ranf((5, 5, 2)) # it takes a tuple as an arguement and size parameter. # this function created random values in an array which are nearer to 0 or even 0 but cannot be 1. only positive values are generated
arr = np.random.randint(1, 100, 200) # it takes 3 arguements. first: min val, second: max val, third: total number of vals. this functions basically generates random intregal numbers between min and max.


'''Creating datatypes in numpy'''
arr = np.arange(11, dtype = np.float32) # using the dtype parameter
arr = np.arange(11, dtype = "i") # using some codes of datatypes like f for float and i for integer. must be encoded in a string.
newarr = np.float32(arr) # creating a new variable and assigning a new datatype to it
anewarr = arr.astype(float) # creating a new variable and assigning a new datatype to it
# newarr and anewarr both work same. the dfference is in newarr it goes by calling numpy and anewarr goes by calling the instance that is the array variable name











'''this is an out concept not of numpy'''
print(print(False) or print(True)) # when both are true and if used and operator, then one one print statement will run and if or operator is used the both the print statements will print
'''and operator checks if both operands are true or not if true then it runs only once.
  or operator runs all operands and runs only those which returns true and ignores falses.
  here print statement will print once when and is used. and will print both prints when or used.
  please remember this concept.
'''


