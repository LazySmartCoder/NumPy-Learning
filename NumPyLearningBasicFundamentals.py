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

arr = np.eye(5, 5, k = 1, dtype = int) # This function creates a diagram of 1. of specified rows and columns and that K parameter takes the index from where to start default is 0. if the value if k is negative then it add 1 each time eg -2+1=-1+1=0+1=1. so from 1 it will start printing the values for more clarity visit - https://numpy.org/doc/stable/reference/generated/numpy.diag.html

'''Creating some random values in more precise manner in numpy.'''
arr = np.random.rand(5, 5, 2) # This functions generates random numbers from 0 to 1.
arr = np.random.randn(5, 5, 2) # this function generates number which are close to 0 it can be positive or negative as well
arr = np.random.ranf((5, 5, 2)) # it takes a tuple as an arguement and size parameter. # this function created random values in an array which are nearer to 0 or even 0 but cannot be 1. only positive values are generated
arr = np.random.randint(1, 100, 200) # it takes 3 arguements. first: min val, second: max val, third: total number of vals. this functions basically generates random intregal numbers between min and max.


'''Converting datatypes in numpy'''
arr = np.arange(11, dtype = np.float32) # using the dtype parameter
arr = np.arange(11, dtype = "i") # using some codes of datatypes like f for float and i for integer. must be encoded in a string.
newarr = np.float32(arr) # creating a new variable and assigning a new datatype to it
anewarr = arr.astype(float) # creating a new variable and assigning a new datatype to it
# newarr and anewarr both work same. the dfference is in newarr it goes by calling numpy and anewarr goes by calling the instance that is the array variable name

# Some arthmetical functions in numpy
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr1 = np.array([[5, 3, 7], [3, 6, 1]])
print(np.cumsum(arr1)) # this functions add the afterward values like if an array is 1, 2, 3 then the out will be 1, 3, 6.
print(np.reciprocal(arr1)) # this function reciprocals the values.
'''Above functions takes multiple arguement but works efficiently with one arguement'''
'''cumsum is used for some statistical puposes like mean.'''


'''Broadcasting in numpy'''
arr = np.array([[1, 2, 3]])
arr1 = np.array([[[1, 2], [2, 5], [3, 9]]])
print(arr1[0, 2, 0])


'''Doing list slicing using 3 shape materials like (1, 2, 3). For more details watch https://youtu.be/-TYSM0CDA4c?t=9053'''
arr1 = np.array([[[1, 2], [2, 5], [3, 9]]])
print(arr1[0, 2, 0])


'''Iterating numpy arrays in numpy'''
arr = np.arange(1, 10).reshape(3, 3)
for i in np.nditer(arr):
  print(i) # or we can also use nested loops as python

# Using the enumerate function of python
arr = np.arange(1, 10).reshape(3, 3)
for x, y in np.ndenumerate(arr):
  print(x, y)

# Using the ndenumerate function of numpy which gives us the indexing
arr = np.arange(1, 10).reshape(3, 3)
for x, y in enumerate(arr):
  print(x, y)


'''Copy vs View in numpy'''
arr = np.array([1, 2, 6, 8, 8])
v = arr.view()
c = arr.copy()
print(id(arr))
print(id(v))
print(id(c))
'''When we use a new variable to assign to an array then the memory location is not effected but it is affected if we use view func. view function has some circumstances which allow to change the original array as well. think more deeply.'''

'''Join split in numpy.'''
arr1 = np.array([[1, 2, 34], [4, 5, 9]])
arr2 = np.array([[3, 6, 54], [4, 8, 10]])
print(np.concatenate((arr1, arr2), axis = 1))
print(arr1)
print(arr2)
# asix1 = column and axis = 0 is row
# column vertical and row horizontal

# hstack: axis=0 and vstack: axis=1. these both functions works same as axis.
# array = row, elements = column. by anirban bhattacharya
arr1 = np.array([[1, 2, 34], [4, 5, 9]])
arr2 = np.array([[3, 6, 54], [4, 8, 10]])
print(np.dstack((arr1, arr2))) # this basically creates an array of indices which matches to indices of other array. it basically prints height wise.
'''Splitting'''

arr1 = np.array([[1, 2, 34], [4, 5, 9]])
print(arr1)
print(np.array_split(arr1, 3, axis=1))
for i in np.array_split(arr1, 3, axis=1):
  print(i)
# this splits the array into two arrays. but be very careful while selecting the split numbers


'''Searching in numpy array'''
arr = np.array([0, 2, 9, 4, 6, 7, 10])
print(np.where(arr!=2)) # this function works on a condition provided.
print(sorted(arr))
print(np.searchsorted(arr, [10, 101, 6], side = "right"))
print(np.searchsorted(arr, [10, 101, 6], side = "left"))
'''The above function do sorts an array and then tells us the indexes that where a particular number can be settled. it can do with right side or left side as well.'''


arr = np.array([[[[0, 2], [9, 4], [6, 7]]]]) # this is a 4d array.
print((arr[0, 0, 0, 1]==2)) # this array can be sorted in this manner or can even be checked.

'''Sorting in numpy'''
arr = np.array([[[[0, 2], [9, 4], [6, 7]]]])
print(np.sort(arr)) # sorting an array in increasing order.
print(np.sort(arr)[0, 0, ::-1]) # sorting in descedning order





'''this is an out concept not of numpy'''
print(print(False) or print(True)) # when both are true and if used and operator, then one one print statement will run and if or operator is used the both the print statements will print
'''and operator checks if both operands are true or not if true then it runs only once.
  or operator runs all operands and runs only those which returns true and ignores falses.
  here print statement will print once when and is used. and will print both prints when or used.
  please remember this concept.
'''


