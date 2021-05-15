#!/usr/bin/env python
# coding: utf-8

# 

# # 102 - Matrices and Matrix Multiplication
# 
# 
# 
# <img src="https://www.mathsisfun.com/algebra/images/matrix-multiply-a.svg" alt="Image showing how matrix multiply works.  There is a 2 by 3 matrix multiplied by a 3 by 2 matrix to get a 2 by 2 matrix.  The first row in the first matrix is highlighted and the first column of the second matrix is highlighted. The words 'Dot Product' are pointing to the highlighted row and column and the single value output is shown in as the only element in the upper left of the 2 by 2 result.  Basically the image is showing that the row [1,2,3] dotted with the column [7,9,11] results in the single output of 58.">
# 
# Image from: [www.mathsisfun.com](https://www.mathsisfun.com/algebra/matrix-multiplying.html)

# ### Goals for today's in-class assignment 
# 
# 1. [Matrix Indexing](#Indexing)
# 1. [Dot Product Review](#Dot-Product-Review)
# 1. [Matrix Multiply](#Matrix-Multiply)
# 1. [Special Examples](#special_matrices)
#     1. [Identity Matrix](#Identity-Matrix)
#     1. [Elementary Matrices](#Elementary-Matrices)
#     1. [Inverse Matrix](#Inverse)
# 1. [Application: Color Transformations](#applications)
# 
# 
# 

# ---- 
# <a name='Indexing'></a>
# ## 1. Matrix Definitions and Indexing
# 
# As we mentioned in our previous class a **_matrix_** is a rectangular array of numbers typically written between rectangular brackets such as:
# 
# $$ A = 
# \left[
# \begin{matrix}
#     0 & -1 \\ 
#     3 & 4 \\
#     0 & 2
#  \end{matrix}
# \right]
# $$
# 
# This rectangular array has three rows and two columns. The size of a matrix is always written $m \times n$ where $m$ is the number of rows and $n$ is the number of columns.  So in the above case Matrix $A$ is a $3 \times 2$  (read "three by two") matrix. 
# 
# If we want to discuss the entries of a matrix then we need to use the specific index of the entry. This is typically given by the ordered pair describing which row and column the entry is in. For example the (2,2)-entry of the matrix $A$ given above is 4. We can write $A_{2,2} =4$, or sometime we will use lowercase letters for the entry: $a_{2,2} = 4$. As a whole matrix we have 
# 
# $$A = 
# \left[
# \begin{matrix}
#     a_{1,1} & a_{1,2} \\ 
#     a_{2,1} & a_{2,2} \\
#     a_{3,1} & a_{3,2}
#  \end{matrix}
# \right].$$
# 
# However with most computer languages the indexing starts with 0 and not 1. Therefore to get the (2,2)-entry of the matrix $A$ in python we need to subtract 1 from all our indices. In general when we discuss specific entries it should be clear from the context which type of indexing we will be using.

# In[1]:


A = [[0,-1],[3,4],[0,2]]

#The entry in the second row, second column (when using a list of lists)
a22 = A[1][1]
a22


# If we use a numpy matrix representation in python then the indexing is slightly different.

# In[2]:


import numpy as np

A_np = np.matrix(A)
print(A_np)

#The entry in the second row, second column (when using a numpy matrix)
a22_np = A_np[1,1]
a22_np


# As a last comment in this section it will also be helpful throughout the semester to select entire rows, or entire columns of a matrix. It is always easy to grab a row of a matrix, but when using the lists of lists representation of a matrix it is hard to grab a column. This is one of many reasons we will transition to using numpy in the rest of the class.

# In[3]:


## The 3rd row of the matrix A:
row3 = A[2]
row3_np = A_np[2]
row3,row3_np


# In[4]:


## The first column of matrix A:
col1_np = A_np[:,0]
col1_np


# &#9989;  **<font color=red>Question</font>**: From the computers point of view what is the difference between ``row3`` and ``row3_np`` given above?

# Type your answer here.

# ### Transpose of  a matrix
# 
# One helpful operator that we will use often in this class is called the the **transpose of a matrix**. If $A$ is a  $m \times n$ matrix then the **transpose of $A$**, denoted by $A^\top$, is the $n \times m$ matrix obtained from $A$ by switching the rows and columns. 
# 
# For example if 
# 
# $$ A = 
# \left[
# \begin{matrix}
#     0 & -1 \\ 
#     3 & 4 \\
#     0 & 2
#  \end{matrix}
# \right], \quad \text{ then } \quad A^\top= 
# \left[
# \begin{matrix}
#     0 & 3 & 0 \\ 
#     -1 & 4 & 2\\ \end{matrix}
#     \right].
# $$
# 
# The transpose is a simple function built into numpy. 

# In[5]:


## Returns the transpose of a matrix.
A_np.T


# 
# ---
# <a name=Dot-Product-Review></a>
# ## 2. Dot Product Review
# 
# 
# 
# 
# 
# We covered inner products yesterday.  This assignment will extend the idea of inner products to matrix multiplication. As a reminder, **_Sections 1.4_** of the [Stephen Boyd and Lieven Vandenberghe Applied Linear algebra book](http://vmls-book.stanford.edu/) covers the dot product.  Here is a quick review:

# Given two vectors $u$ and $v$ in $R^n$ (i.e. they have the same length), the "dot" product operation multiplies all of the corresponding elements  and then adds them together. Ex:
# 
# $$u = [u_1, u_2, \dots, u_n]$$
# $$v = [v_1, v_2, \dots, v_n]$$
# 
# $$u \cdot v = u_1 v_1 + u_2  v_2 + \dots + u_nv_n$$
# 
# or:
# 
# $$ u \cdot v = \sum^n_{i=1} u_i v_i$$
# 
# &#9989;  **<font color=red>Do This</font>**: Find the dot product of the vectors $u = (1,2,3)$ and $v = (7,9,11)$.

# In[6]:


##Do your work here


# In[7]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer.float(dot,'9ed469ac3b8ef2d21d85e191c8cd24cd');


# 
# 
# ---
# <a name=Matrix-Multiply></a>
# ## 3. Matrix Multiply
# 
# Two matrices $A$ and $B$ can be multiplied together if and only if their "inner dimensions" are the same, i.e. $A$ is 
# $m\times d$ and $B$ is $d\times n$ (note that the columns of $A$ and the rows of $B$ are both $d$). 
# Multiplication of these two matrices results in a third matrix $C$ with the dimension of $m\times n$.
# Note that $C$ has the same first dimension as $A$ and the same second dimension as $B$. i.e $m\times n$.  
# 
# _**The $(i,j)$ element in $C$ is the dot product of the $i$th row of $A$ and the $j$th column of $B$.**_
# 
# The $i$th row of $A$ is:
# 
# $$ [ a_{i1},  a_{i2},  \dots , a_{id} ],$$
# 
# and the $j$th column of $B$ is:
# 
# $$
# \left[
# \begin{matrix}
#     b_{1j}\\ 
#     b_{2j}\\
#     \vdots \\
#     b_{dj}
# \end{matrix}
# \right] 
# $$
# 
# So, the dot product of these two vectors is:
# 
# $$c_{ij} = a_{i1}b_{1j} + a_{i2}b_{2j} + \dots + a_{id}b_{dj}$$
# 
# Consider the simple $2\times 2$ example of matrix multiplication given below:
# 
# 
# $$ 
# \left[
# \begin{matrix}
#     a & b\\ 
#     c & d   
# \end{matrix}
# \right] 
# \left[
# \begin{matrix}
#     w & x\\ 
#     y & z   
# \end{matrix}
# \right] 
# =
# \left[
# \begin{matrix}
#     aw+by & ax+bz\\ 
#     cw + dy & cx + dz   
# \end{matrix}
# \right] 
# $$
# 
# For more information read **_Section 10.1_** of the [Stephen Boyd and Lieven Vandenberghe Applied Linear algebra book](http://vmls-book.stanford.edu/) which covers Matrix Multiplication.

# Let's do an example using ```numpy``` and show the results using ```sympy```:

# In[ ]:


import numpy as np
import sympy as sym
sym.init_printing(use_unicode=True) # Trick to make matrixes look nice in jupyter


# In[ ]:


A = np.matrix([[1,1], [2,2]])
sym.Matrix(A)


# In[ ]:


B = np.matrix([[3,4], [3,4]])
sym.Matrix(B)


# In[ ]:


sym.Matrix(A*B)


# &#9989;**<font color=red>DO THIS</font>**: Given two matrices; $A$ and $B$, show that order matters when doing a matrix multiply. That is, in general, $AB \neq BA$. 
# Show this with an example using two $3\times 3$ matrices and ```numpy```.

# In[ ]:


# Put your code here.


# &#9989;**<font color=red>QUESTION</font>**: What is the size of the matrix resulting from multiplying a $10 \times 40$ matrix with a $40 \times 3$ matrix?

# Put your answer here

# 
# ---
# ### List Implementation
# 
# Now that we have an understanding of how matrix multiplication works. Use the definitions given above to implement your own matrix multiplication function in python. Youshould use the list of lists format and not use any numpy functions. 
# 
# &#9989; **<font color=red>DO THIS:</font>** Write your own matrix multiplication function using the template below and compare it to the built-in matrix multiplication that can be found in ```numpy```. Your function should take two "lists of lists" as inputs and return the result as a third list of lists.  

# In[ ]:


#some libraries (maybe not all) you will need in this notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import sympy as sym
sym.init_printing(use_unicode=True)

import random
import time


# In[ ]:


def multiply(m1,m2):
    #first matrix is nxd in size
    #second matrix is dxm in size
    n = len(m1) 
    d = len(m2)
    m = len(m2[0])
    
    #check to make sure sizes match
    if len(m1[0]) != d:
        print("ERROR - inner dimentions not equal")
    
    #### put your matrix multiply code here #####
    
    return result


# Test your code with the following examples

# In[ ]:


#Basic test 1
n = 3
d = 2
m = 4

#generate two random lists of lists.
matrix1 = [[random.random() for i in range(d)] for j in range(n)]
matrix2 = [[random.random() for i in range(m)] for j in range(d)]


# In[ ]:


sym.init_printing(use_unicode=True) # Trick to make matrixes look nice in jupyter

sym.Matrix(matrix1) # Show matrix using sympy


# In[ ]:


sym.Matrix(matrix2) # Show matrix using sympy


# In[ ]:


#Compute matrix multiply using your function
x = multiply(matrix1, matrix2)


# In[ ]:


#Compare to numpy result
np_x = np.matrix(matrix1)*np.matrix(matrix2)

#use allclose function to see if they are numrically "close enough"
print(np.allclose(x, np_x))

#Result should be True


# In[ ]:


#Test identity matrix
n = 4

# Make a Random Matrix
matrix1 = [[random.random() for i in range(n)] for j in range(n)]
sym.Matrix(matrix1) # Show matrix using sympy


# ### Timing Study
# In this part, you will compare your matrix multiplication with the ```numpy``` matrix multiplication. 
# You will multiply two randomly generated $n\times n$ matrices using both the ```multiply()``` function defined above and the ```numpy``` matrix multiplication. 
# Here is the basic structure of your timing study:
# 
# 1. Initialize two empty lists called ```my_time``` and ```numpy_time```
# 2. Loop over values of n (100, 200, 300, 400, 500)
# 3. For each value of $n$ use the time.clock() function to calculate the time it takes to use your algorithm and append that time (in seconds) to the ```my_time``` list.
# 4. For each value of $n$ use the time.clock() function to calculate the time it takes to use the ```numpy``` matrix multiplication and append that time (in seconds) to the ```numpy_time``` list.
# 5. Use the provided code to generate a scatter plot of your results.

# In[ ]:


n_list = [100, 200, 300, 400, 500]
my_time = []
numpy_time = []


# In[ ]:


# # RUN AT YOUR OWN RISK.
# # THIS MAY TAKE A WHILE!!!!

# for n in n_list:
#     print(f"Measureing time it takes to multiply matrixes of size {n}")
#     #Generate random nxn array of two lists
#     matrix1 = [[random.random() for i in range(n)] for j in range(n)]
#     matrix2 = [[random.random() for i in range(n)] for j in range(n)]
#     start = time.time()
#     x = multiply(matrix1, matrix2)
#     stop = time.time()
#     my_time.append(stop - start)
    
#     #Convert the lists to a numpy matrix
#     npm1 = np.matrix(matrix1)
#     npm2 = np.matrix(matrix2)

#     #Calculate the time it takes to run the numpy matrix. 
#     start = time.time()
#     answer = npm1*npm2
#     stop = time.time()
#     numpy_time.append(stop - start)


# In[ ]:


# plt.scatter(n_list,my_time, color='red', label = 'my time')
# plt.scatter(n_list,numpy_time, color='green', label='numpy time')

# plt.xlabel('Size of $n x n$ matrix');
# plt.ylabel('time (seconds)')
# plt.legend();


# Based on the above results, you can see that the ```numpy``` algorithm not only is faster but also "scales" at a slower rate than your algorithm.  

# &#9989; **<font color=red>QUESTION:</font>** Why do you think the ```numpy``` matrix multiplication is so much faster?  

# Put your answer to the above question here

# 
# ---
# <a name='special_matrices'></a>
# ## 4. Examples of special matrices
# 
# The following matrices have special properties when it comes to matrix multiplication.
# 
# <a name=Identity-Matrix></a>
# ### Identity Matrix
# 
# A matrix is said to be **square** if it has the same number of rows and columns.
# An identity matrix is a special square matrix (i.e. $m=n$) that has ones in the diagonal and zeros other places. For example the following is a $3\times 3$ identity matrix:
# 
# $$
# I_3 = 
# \left[
# \begin{matrix}
#     1 & 0 & 0\\ 
#     0 & 1 & 0 \\
#     0 & 0 & 1
# \end{matrix}
# \right] 
# $$
# 
# We always denote the identity matrix with a capital $I$. Often a subscript is used to denote the value of $n$. The notations $I_{nxn}$ and $I_n$ are both acceptable.
# 
# An identity matrix is similar to the number 1 for scalar values.  I.e. multiplying a square matrix $A_{nxn}$ by its corresponding identity matrix $I_{nxn}$ results in itself $A_{nxn}$.
# 
# &#9989;**<font color=red>DO THIS</font>**: Pick a random $3\times 3$ matrix and multiply it by the $3\times 3$ Identity matrix and show you get the same answer. 

# In[ ]:


#Put your code here


# &#9989; **<font color=red>QUESTION:</font>** Consider two square matrices $A$ and $B$ of size $n \times n$.  $AB = BA$ is **NOT** true for many $A$ and $B$.  Describe an example where $AB = BA$ is true? Explain why the equality works for your example.

# Put your answer here

# 
# ---
# <a name=Elementary-Matrices></a>
# ### Elementary Matrices
# 
# 
# **_NOTE_**: A detailed description of elementary matrices can be found here in the **_Beezer text Subsection EM 340-345_** if you find the following confusing. 
# 
# There exist a cool set of matrices that can be used to implement Elementary Row Operations. Elementary row operations include:
# 
# 1. Swap two rows
# 2. Multiply a row by a constant ($c$)
# 3. Multiply a row by a constant ($c$) and add it to another row.
# 
# You can create these elementary matrices by applying the desired elementary row operations to the identity matrix. 
# 
# If you multiply your matrix from the left using the elementary matrix, you will get the desired operation.
# 
# For example, here is the elementary row operation to swap the first and second rows of a $3\times 3$ matrix:
# 
# $$ 
# E_{12}=
# \left[
# \begin{matrix}
#     0 & 1 & 0\\ 
#     1 & 0 & 0 \\
#     0 & 0 & 1
# \end{matrix}
# \right] 
# $$
# 

# In[ ]:


import numpy as np
import sympy as sym
sym.init_printing(use_unicode=True)
A = np.matrix([[3, -3,9], [2, -2, 7], [-1, 2, -4]])
sym.Matrix(A)


# In[ ]:


E1 = np.matrix([[0,1,0], [1,0,0], [0,0,1]])
sym.Matrix(E1)


# In[ ]:


A1 = E1*A
sym.Matrix(A1)


# &#9989;  **<font color=red>DO THIS</font>**: Give a $3\times 3$ elementary matrix named ```E2``` that swaps row 3 with row 1 and apply it to the $A$ Matrix. Replace the matrix $A$ with the new matrix.

# In[ ]:


# Put your answer here.  
# Feel free to swich this cell to markdown if you want to try writing your answer in latex.


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(E2,'2c2d2e407389eabeb6d90894565c830f');


# &#9989;  **<font color=red>DO THIS</font>**: Give a $3\times 3$ elementary matrix named ```E3``` that multiplies the first row by $c=3$ and adds it to the third row. Apply the elementary matrix to the $A$ matrix. Replace the matrix $A$ with the new matrix.

# In[ ]:


# Put your answer here.  
# Feel free to swich this cell to markdown if you want to try writing your answer in latex.


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(E3,'55ae1f9eb21df00c59dad623b9471506');


# &#9989;  **<font color=red>DO THIS</font>**: Give a $3\times 3$ elementary matrix named ```E4``` that multiplies the second row by a constant $c=1/2$ applies this to matrix $A$.

# In[ ]:


# Put your answer here.  
# Feel free to swich this cell to markdown if you want to try writing your answer in latex.


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(E4,'3a5256840ef907a1b73ebba4471ac26d');


# If the above are correct then we can combine the three operators on the original matrix $A$ as follows. 

# In[ ]:


A = np.matrix([[3, -3,9], [2, -2, 7], [-1, 2, -4]])

sym.Matrix(E4*E3*E2*A)


# ---
# <a name='Inverse'></a>
# ### Inverse Matrices
# 
# For some (not all) **square** matrices $A$, there exists a special matrix called the Inverse Matrix, which is typically written as $A^{-1}$ and when multiplied by $A$ results in the identity matrix $I$:
# 
# $$ A^{-1}A = AA^{-1} = I $$
# 
# We say that a matrix $A$ is **invertible** if there exists an inverse matrix $A^{-1}$ such that the above equalities hold. 
# 
# Some properties of an Inverse Matrix include:
# 
# 1. $(A^{-1})^{-1} = A$
# 2. $(cA)^{-1} = \frac{1}{c}A^{-1}$
# 3. $(AB)^{-1} = B^{-1}A^{-1}$
# 4. $(A^n)^{-1} = (A^{-1})^n$
# 1. $(A^\top)^{-1} = (A^{-1})^\top$  here $A^\top$ is the tranpose of the matrix $A$.
# 
# 
# &#9989;  **<font color=red>DO THIS:</font>** Find a Python numpy command that will calculate the inverse of a matrix and use it invert the following matrix ```A```.  Store the inverse in a new matirx named ```A_inv```

# In[ ]:


import numpy as np
import sympy as sym
sym.init_printing(use_unicode=True) # Trick to make matrixes look nice in jupyter

A = np.matrix([[1, 2, 3], [4, 5, 6], [7,8,7]])

sym.Matrix(A)


# In[ ]:


#put your answer to the above question here.


# Lets check your answer by multiplying ```A``` by ```A_inv```. 

# In[ ]:


A * A_inv


# In[ ]:


np.allclose(A*A_inv, [[1,0,0],[0,1,0],[0,0,1]])


# ### How do we create an inverse matrix?
# 
# From previous assignments, we learned that we could string together a bunch of Elementary Row Operations to get matrix ($A$) into it's Reduced Row Echelon form. We now know that we can represent Elementary Row Operations as a sequence of Elementaary Matrices as follows:
# 
# $$ E_n \dots E_3 E_2 E_1 A = I $$
# 
# If $A$ reduces to the identity matrix (i.e. $A$ is row equivalent to $I$), then $A$ has an inverse and its inverse is just all of the Elementary Matrices multiplied together:
# 
# $$ A^{-1} = E_n \dots E_3 E_2 E_1 $$
# 
# Consider the following matrix.  
# $$
# A = \left[
# \begin{matrix}
#     1 & 2 \\ 
#     4 & 6 
# \end{matrix}
# \right] 
# $$

# In[ ]:


A = np.matrix([[1, 2], [4,6]])


# It can be reduced into an identity matrix using the following elementary operators
# 
# | Words | Elementary Matrix|
# |:---:|:---:|
# | Adding -4 times row 1 to row 2. | $$E_1 = \left[\begin{matrix}1 & 0 \\ -4 & 1 \end{matrix}\right]$$ |
# |Adding row 2 to row 1. |$$
# E_2 = \left[
# \begin{matrix}
#     1 & 1 \\ 
#     0 & 1 
# \end{matrix}
# \right] $$ |
# | Multiplying row 2 by $-\frac{1}{2}$.| $$
# E_3 = 
# \left[
# \begin{matrix}
#     1 & 0 \\ 
#     0 & -\frac{1}{2} 
# \end{matrix}
# \right]
# $$ |

# In[ ]:


E1 = np.matrix([[1,0], [-4,1]])
E2 = np.matrix([[1,1], [0,1]])
E3 = np.matrix([[1,0],[0,-1/2]])


# We can just check that the statment seems to be true by multiplying everything out.

# In[ ]:


E3*E2*E1*A


# &#9989;  **<font color=red>DO THIS:</font>** Combine the above elementary Matrices to make an inverse matrix named ```A_inv```

# In[ ]:


# Put your answer to the above question here.


# &#9989;  **<font color=red>DO THIS:</font>** Verify that ``A_inv`` is an actual inverse and chech that $AA^{-1} = I$.

# In[ ]:


# Put your code here.


# Notice, at the beginning of the section we said that not all matrices are invertible. For starters we require that a matrix be square before we can even begin to discuss if it is invertible. An example of a matrix that does not have an inverse is
# 
# $$B = \begin{bmatrix}
#  1 & 3 \\ 
#  2 & 6 \\
# \end{bmatrix}.$$
# 
# We will discuss ways to prove this fact as the course develops.
# 

# ----
# <a name='applications'></a>
# ## Application: Changing Color Vectors
# 
# We will study applications of matrix multiplication to transforming points in space and robotics next week in assignment 104, but for today we will now look at changing the colors in a digital image.
# 
# In the first class we talked about how a computer stores a color as a 3-vector representing the amount of red, green, and blue to add to a pixel. We saw briefly in the pre-class assignment how a computer stores a digital image. Specifically, they are stored as a triple $(x,y,C)$ where $x$ and $y$ represent the coordinates for the pixel in the image, and $C$ is a color vector representing the color of the pixel at that coordinate. 
# 
# What we do in the following is start with an image which is $h$-pixels tall and $w$-pixels wide, and lay out all of color vectors horizontally into a $ 3 \times hw$ matrix. Then we multiply this new matrix by a $3 \times 3$ matrix which will transform the image into a new one. 
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import sympy as sym
sym.init_printing()


# In[ ]:


import imageio

## If neither of the url's below work for you please change it to any image url (use google). 
## Note however, that the smaller the dimensions of the image the faster the code will run.
url = 'http://www.ideachampions.com/weblogs/iStock_000022162723_Small.jpg'
#url = 'http://colortutorial.design/rgb.jpg'
im = imageio.imread(url)
plt.imshow(im)


# The functions below will convert between images represented by 3-dimensional arrays of shape $(h,w,3)$, and numpy matrices of size $3 \times n$. In the $3 \times n$ matrix each column represents the color of an individual pixel of our image.

# In[ ]:


def image2matrix(image):
    '''Turns an image into a numpy matrix of size 3 x (h*w).'''
    h,w,_ = image.shape
    return np.matrix(image.reshape(h*w,3)).T

def matrix2image(matrix,h,w):
    '''Turns a 3xn matrix into a numpy array of shape (h,w,3).'''
    
    if h*w != matrix.shape[1]:
        raise Exception("Matrix must have length of h*w!")
        
    return np.array(matrix.T).reshape(h,w,3).astype(int).clip(0,255)


# Below is the image matrix that we will be using for our examples. It is saved as ``im_mat``.

# In[ ]:


h,w,_ = im.shape
im_mat = image2matrix(im)
im_mat.shape


# If we apply a matrix multiplication to ``im_mat`` we are changing the color vectors associated to the pixel.
# The idea is that if we have a matrix $M$ and a color vector $x = (R,G,B)^T$, then $Mx$ will be a new $ 3 \times 1$ matrix that represents a new color. 

# In[ ]:


sepia = np.matrix('0.393,0.769,0.189;0.349,0.686,0.168;0.272,0.534,0.131')
sepia_mat = sepia * im_mat
plt.imshow(matrix2image(sepia_mat,h,w))


# &#9989;  **<font color=red>Do This:</font>** (5pts) Apply the following grayscale transformation below to the image matrix ``im_mat``.

# In[ ]:


## Edit this cell for your work. 
grayscale = np.matrix(np.ones([3,3]))/3

#new_image_matrix = 


# In[ ]:


##This will show the transformed image.
plt.imshow(matrix2image(new_image_matrix,h,w))


# &#9989;  **<font color=red>QUESTION 4:</font>** (5pts) Create a $3 \times 3$ elementary matrix ```E``` that swaps the red and blue values in a color vector.
# That is we want ``E`` to be a matrix such that $Ex = x'$ where $x = [R,G,B]^T$ and $x' = [B,G,R]^T$. 

# In[ ]:


## Edit this cell for your work. 


# In[ ]:


plt.imshow(matrix2image(A*im_mat,h,w))


# 
# 
