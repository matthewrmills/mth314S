#!/usr/bin/env python
# coding: utf-8

# 

# In order to successfully complete this assignment you need to participate both individually and in groups during class.   If you attend class in-person then have one of the instructors check your notebook and sign you out before leaving class. If you are attending asynchronously, turn in your assignment using D2L no later than **_11:59pm on the day of class_**. See links at the end of this document for access to the class timeline for your section.

# # 101 - In-class Assignment: Welcome and Introduction to Vectors
# 
# 1. [Textbooks & Syllabus](#books)
# 1. [Making sure Jupyter works](#jupyter)
# 1. [Scalars, Vectors, Matrices](#vectors)
#     1. [Definitions](#math_def)
#     1. [Basic Python](#basic_python)
# 1. [Vector Examples](#examples)
# 1. [Vector Addition and Scalar Multiplication](#add)
# 1. [Linear combinations](#linear_combs)

# ----
# <a name="books"></a>
# ## 1. Text Books & Syllabus

# The textbooks for this course are all online and available for download. 

# Student self guided learning through assigned readings are required for students to be successful.  The course strives to use Open Educational Resources (OER) to help reduce financial burden on the students.  To this end we have selected the following textbooks for reading assignments and supplemental examples:  
# 
# 
# * [Introduction to Applied Linear Algebra](http://vmls-book.stanford.edu/) by Boyd and Vandenberghe
# * [Linear Algebra](http://joshua.smcvt.edu/linearalgebra/) by Jim Heffron
# * [A First Course in Linear Algebra](http://linear.ups.edu/) by Robert A. Beezer
# 
# 
# **_DO NOT WORRY_** You will not be expected to read all three textbooks in this course!  In fact, we try to keep the reading at a reasonable level and focus on problem solving.  However, most students benefit from seeing material in multiple ways (Reading, Lecture, Practice, etc).  Different students (and instructors) also prefer different writing styles and may learn better with different text (which is why we provide more than one).  
# 
# Students are encouraged to review and become familiar with the style and layout of each text to act as a common reference for the course.  If you get stuck on a topic try looking it up and reviewing it in one of the other texts.  If you are still stuck you can search the Internet.  Do not be afraid to also ask your instructors questions and come to office hours. That is why we are here!!!
# 
# &#9989; **<span style="color:red">Do This:</span>** Download a copy of each textbooks onto your preferred reading device and review the **_Table of Contents_** in each text.
# 
# 
# As you can see each textbook approaches Linear algebra in a slightly different way. This variety reflects the different philosophies of teaching and different ways of individual learning. One way to evaluate the focus of a particular textbook is to look at it's very first chapter.  For Example:
# 
# * The **_Beezer_** and **_Heffron_** texts start out with "Systems of linear Equations" and "Linear Systems." These topics are basically the same idea with the focus of defining linear systems as just sets of "linear combinations".  Clearly this is a core concept and a good place to start.
# * The **_Boyd and Vandenberghe_** text choose to start with "Vectors".  In linear algebra the "vector" is a mathematical tool for which all of the mechanics of the math is built.  Again, not a bad place to start. 
# 
# In the first few assignments this course we will be looking at both concepts.  You will want to learn and be able to identify linear systems and how to represent them as vectors. 

# In[1]:


## If you uncomment the text below this cell will download the pdf
## and put it in your currect directory (not required)

# from urllib.request import urlretrieve
# urlretrieve('http://vmls-book.stanford.edu/vmls.pdf',  'Boyd_and_Vandenberghe.pdf');
# urlretrieve('http://joshua.smcvt.edu/linearalgebra/book.pdf',  'Heffron.pdf');
# urlretrieve('http://linear.ups.edu/download/fcla-3.50-tablet.pdf',  'Beezer.pdf');


# ### Syllabus
# 
# Please review the course [syllabus](https://docs.google.com/document/u/1/d/e/2PACX-1vRXcy56SikBnJ9pWOpujibVtfC0VTyNCtXp1I11MZ4CVhJg9hpqCbFYv0T77VOEyM0dlyYxzrCHpUJf/pub). This link can be permantly found on D2L.

# ---
# <a name="jupyter"></a>
# ## 2. Making sure Jupyter works
# 
# If you are veiwing this file as an html, or pdf please follow the instructions in the notebook in order to be able to edit the cells. It is very important to get this working as our entire class will be using this software.
# 
# > [0000--Jupyter-Getting-Started-Guide](https://msu-cmse-courses.github.io/mth314-F20-student/assignments/0000--Jupyter-Getting-Started-Guide.html)
# 
# The instructions to set up Jupyter are also given at the end of the syllabus.

# ----
# <a name="vectors"></a>
# ## 3. Scalars, Vectors, and Matrices
# 
# <a name="math_def"></a>
# ### Mathematical Definitions
# 
# The three primary mathematical entities that are of interest in this class are scalars, vectors, and matrices. 
# 
# Think of a **_scalar_** as a single number or variable. The following are all scalars:
# 
# $$ 1, \frac{1}{2}, 3.1416$$
# 
# A **_vector_**, on the other hand, is an *ordered* list of values which we typically represent with lower case letters. Vectors are ordered arrays of single numbers and are typically written as a row or a column. The following are all vectors:
# 
# **_Row Vector:_** 
# $$v = [ v_1, v_2, \dots, v_n ]$$
# here $v_1, v_2, \dots, v_n$ are single numbers.
# 
# $$f = [1, 2, 3, 5, 8]$$
# 
# Here $f$ in the above example is a vector of numbers, and it is the common way we think of vectors.
# 
# Note, it is more common in this class to write vectors vertically. These are often called column vectors:
# 
# **_Column Vector:_**
# $$
# v=
# \left[
# \begin{matrix}
#     v_1 \\ 
#     v_2 \\
#     \vdots \\
#     v_m
#  \end{matrix}
# \right]
# $$
# here $v_1, v_2, \dots, v_n$ are single numbers.

# A **_matrix_** is a rectangular array of numbers typically written between rectangular brackets such as:
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
# We will use matrices everyday in this class, but won't talk about them in depth untill next class. 
# 
# In essence, you can consider a row-vector to be a $1 \times n$ matrix, or a column-vector to be an $m \times 1$ matrix.

# -----
# <a name="basic_python"></a>
# ### Basic Python Structures
# 
# Defining a **_scalar_** in Python is easy. For example

# In[2]:


a = 8
a


# In Python, there are multiple ways to store a vector. Knowing how your vector is stored is very important (especially for debugging). Probably the easiest way to store a vector is using a list, which are created using standard square brackets as follows:

# In[3]:


f = [1, 2, 3, 5, 8]
f


# Another common way to store a vector is to use a tuple.  

# In[4]:


b = (2, 4, 8, 16)
b


# You can access a particular scalar in your Python object using its index. Remember that Python index starts counting at zero. For example, to get the fourth element in ```f``` and ```b``` vectors, we would use the following syntax:

# In[5]:


print(f[3])
print(b[3])


# Later in this course, we may discuss which data format is better (and introduce new ones). At this point let's not worry about it too much. You can always figure out a variable's data type using the ```type``` function. For example:type(f)

# In[6]:


type(f)


# In[7]:


type(b)


# Finally, I am not sure if you will need this but always remember, it is easy to convert from a tuple to a list and vice versa (this is called "casting"):

# In[8]:


#Convert tuple to list
b_list = list(b)
b_list


# In[9]:


#Convert list to tuple
f_tuple = tuple(f)
f_tuple


# A vector can be used to represent quantities or values in an application. The size (also called dimension or length) of the vector is the number of elements it contains.  The size of the vector determines how many quantities are in the vector.  We often refer to the size of a vector using the variable ```n```.  So an ```n-vector``` has ```n``` values.  A ```3-vector``` only has 3 values. 
# 
# The length (```len```) function returns the size of a vector in Python:

# In[10]:


len(f)


# In[11]:


len(b)


# A matrix can be stored as a list of lists, where the inside lists are the row-vectors of the matrix. For example the matrix 
# 
# $$ M = \begin{bmatrix}2 & 3 & 0 \\ 1 & -4 & -1\end{bmatrix} $$
# 
# can be saved as below:

# In[12]:


M = [[2,3,0],
     [1,-4,-1]]
M


# Naturally, in this format we can find the shape of the matrix again with the ``len`` function. Recall, that a $ m \times n $ matrix has $m$ rows and $n$ columns. Since the inner lists are row-vectors we know that the length of the row is the number of columns.

# In[13]:


m = len(M)
n = len(M[0])
print(f'The matrix M has {m} rows and {n} columns.')


# ### Syntax Errors
# A syntax error means that the code does not make sense in Python. We would like to define a couple of vectors with four numbers.

# &#9989; **<font color=red>DO THIS:</font>** Fix the following code to creat three vectors with four numbers.

# In[14]:


x = [1 2 3.4 4]
y = [1, 2, 3, 5]]
z = [[1, 2, 3, 6.3]


# Although you may have been able to get rid of the error messages the answer to you problem may still not be correct.  Throughout the semester we will be using a python program called ```answercheck``` to allow you to check some of your answers.  This program doesn't tell you the right answer but it is intended to be used as a way to get immediate feedback and accelerate learning.

# In[ ]:


from urllib.request import urlretrieve

urlretrieve('https://raw.githubusercontent.com/colbrydi/jupytercheck/master/answercheck.py', 
            'answercheck.py');


# &#9989; **<font color=red>DO THIS:</font>** Just run the following command to see if you got ```x```, ```y``` and ```z``` correct when you fixed the code above. 

# In[ ]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer([x,y,z],'e80321644979a873b273aebbbcd0e450');


# **_NOTE_** make sure you do not change the ```checkanswer``` commands.  The long string with numbers and letters is the secret code that encodes the true answer.  This code is also called the HASH.  Feel free to look at the ```answercheck.py``` code and see if you can figure out how it works?  

# ---
# <a name="examples"></a>
# ## 4. Examples
# 
# 
# Vectors are used to represent all types of data that has structures.  Here are some simple examples from the [Boyd and Vandenberghe textbook](http://vmls-book.stanford.edu/):
# 
# 
# ### Location and displacement
# A 2-vector can be used to represent a position or location in a space.  The first value is the distance in one direction (from the origin) and the second value is the distance in a different direction.  Probably most students are famiar with the 2D Cartesian coordinate system where a location can be defined by two values in the ```x``` and ```y``` directions.  Here is a simple scatter plot in python which show the concept:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
p1 = [2, 1]
p2 = [1, 3]
p3 = [1, 1]

plt.plot(p1[0],p1[1],'*k')
plt.plot(p2[0],p2[1],'*k')
plt.plot(p3[0],p3[1],'*k')

## Add some labels (offset slightly)
plt.text(p1[0]+0.1,p1[1],'$p_1$')
plt.text(p2[0]+0.1,p2[1],'$p_2$')
plt.text(p3[0]+0.1,p3[1],'$p_3$')

## Fix the axis so you can see the points
plt.axis([0,4,0,4]);


# ### Color
# A 3-vector can represent a color, with its entries giving the Red, Green, and Blue (RGB) intensity values (often between 0 and 1). The vector (0,0,0) represents black, the vector (0, 1, 0) represents a bright pure green color, and the vector (1, 0.5, 0.5) represents a shade of pink. 
# 
# The Python ```matplotlib``` library uses this type of vector to define colors.  For example, the following code plots a point at the origin of size 10000 (the size of the circle, and the value does not have exact meaning here) and color c = (0,1,0).  You can change the values for ```c``` and ```s``` to see the difference.

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

c = (0, 1, 0)
plt.scatter(0,0, color=c, s=10000);


# Just for fun, here is a little interactive demo that lets you play with different color vectors. 
# 
# 
# > **_NOTE_** this demo uses the ```ipywidgets``` Python library which works by default in Jupyter notebook (which is installed on the MSU [jupyterhub](http://jupyterhub.egr.msu.edu))) but **_NOT_** in the newer jupyter lab interface which some students may have installed on their local computers.  To get these types of examples working in jupyter lab requires the installation of the ipywidgets plug-in. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
from ipywidgets import interact, fixed

def showcolor(red,green,blue):
    color=(red,green,blue)
    plt.scatter(0,0, color=color, s=20000);
    plt.axis('off');
    plt.show();
    return color

color = interact(showcolor, red=(0.0,1.0), green=(0.0,1.0), blue=(0.0,1.0));


# ---
# <a name="operations"></a>
# ## 5. Vector Addition and Scalar Multiplication
# 
# ### Vector Addition
# 
# Another common operation to perform on vectors is addition. Two vectors of the same size can be added together by adding the corresponding elements, to form another vector of the same size, called the sum of the vectors. For example:
# 
# $$ 
# \left[
# \begin{matrix}
#     1  \\ 
#     20   
#  \end{matrix}
#  \right]
#  +
# \left[
# \begin{matrix}
#     22 \\ 
#     -3 
#  \end{matrix}
#  \right]
#   =
# \left[
# \begin{matrix}
#     23 \\ 
#     17 
#  \end{matrix}
# \right]
# $$
# 
# Here is where things get tricky in Python.  If you try to add a list or tuple, Python does not do the vector addition as we defined above. In the following examples, notice that the two lists concatenate instead of adding by element: 

# In[ ]:


## THIS IS WRONG
a = [1, 20]
b = [22,-3]
c = a+b
c


# In[ ]:


## THIS IS ALSO WRONG
a = (1, 20)
b = (22,-3)
c = a+b
c


# To do proper vector math you need either use a special function (we will learn these) or loop over the list.  Here is a very simplistic example:

# In[ ]:


a = (1, 20)
b = (22,-3)
c = []
for i in range(len(a)):
    c.append(a[i] + b[i])
c


# In[ ]:


def vecadd(a,b):
    """Function to add two equal size vectors."""
    if (len(a) != len(b)):
        raise Exception('Error - vector lengths do not match')
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c


# In[ ]:


#Lets test it
vecadd(a,b)


# ### Scalar Multiplication
# Earlier we defined **_scalar_** as a number. The reason why we call it a "scalar" is because when we multiply a vector by a number the geometric representation of the new vector is still pointing in the same direction, but it has a new length, i.e., it was rescaled.
# 
# If we have the vector $v = \begin{bmatrix} 1 \\ 2\end{bmatrix}$ and multiply it by the scalar 3 then we get a new vector $$3v = 3 \begin{bmatrix} 1 \\ 2\end{bmatrix} =  \begin{bmatrix} 3\cdot 1 \\ 3\cdot 2\end{bmatrix} = \begin{bmatrix} 3 \\ 6\end{bmatrix}$$
# 
# Notice that scalar multiplication (by an integer) is analagous to regular multiplication in that multiplication is just repeated addition, e.g., $3v = v+v+v$.
# 
# More generally, when we multiply a vector $v= [v_1,\dots,v_n]$ by any real scalar $a$ we obtain the vector $av = [a\cdot v_1,\dots,a\cdot v_n]$. 
# 
# When we are using python lists to represent our vectors the multiplication symbol ``*`` is programed to concatenate multiple copies of a list. See below. 

# In[ ]:


##THIS IS WRONG## 
z = 3
a = [3,-7,10]
c = z*a
c


# Again, in order to do proper vector math in Python you need either use a special function (we will learn these) or loop over the list.  
# 
# &#9989; **<span style="color:red">Do This:</span>**  See if you can make a simple function with a loop to multiply a scalar by a vector. Name your function ```sv_multiply``` and test it using the cells below.
# 
# **Note.** You can quickly comment/uncomment code in a cell by using the shortcut ``command + /`` on a mac or ``control + /`` on other machines. For other keyboard shorcuts in Jupiter notebooks if you aren't typing in a cell and press the letter ``H`` a help menu will pop-up.

# In[ ]:


# put your sv_multiply function here

# def sv_multiply(s,v):
#     return 


# In[ ]:


#Test your function here
s = 3
v = [3,-7,10]
sv_multiply(s,v)


# Let us use the following code to test your functon further. 

# In[ ]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer.vector(sv_multiply(10,[1,2,3,4]),'414a6fea724cafda66ab5971f542adf2')


# In[ ]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer.vector(sv_multiply(3.14159,(1,2,3,4)),'f349ef7cafae77c7c23d6924ec1fd36e')


# **Note that scalar multiplication and addition works exactly the same way for matrices.**

# ---
# <a name="linear_combs"></a>
# ## 6. Linear Combinations of vectors
# 
# A linear combination of vectors uses both scalar multiplication and vector addition.
# 
# Let $v_1,\dots,v_n$ be a finite collection of vectors (of the same length) and $a_1,\dots,a_n$ be a finite collection of scalars. The linear combination of the vectors $\left\{v_i\right\}_{i=1}^n$ with the scalars $\left\{a_i\right\}_{i=1}^n$ is the expression 
# 
# $$a_1 v_1 + a_2 v_2 + \dots + a_n v_n.$$
# 
# For example if $a_1 = 3$, $a_2 = -2$, and $a_3 = 1.5$, and $$v_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 1 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \\ -1 \end{bmatrix}, \quad v_3 =\begin{bmatrix} 2 \\ 6 \\ 4 \\ 8 \end{bmatrix}$$ then the corresponding linear combination $3 v_1 -2 v_2 +1.5 v_3$ gives us a new vector:
# 
# $$ 3\begin{bmatrix} 1 \\ 1 \\ 0 \\ 1 \end{bmatrix} -2\begin{bmatrix} 0 \\ 1 \\ 0 \\ -1 \end{bmatrix} +1.5 \begin{bmatrix} 2 \\ 6 \\ 4 \\ 8 \end{bmatrix} \quad=\quad
# \begin{bmatrix} 3 \\ 3 \\ 0 \\ 3 \end{bmatrix} +\begin{bmatrix} 0 \\ -2 \\ 0 \\ 2\end{bmatrix} + \begin{bmatrix} 3 \\ 9 \\ 6 \\ 12 \end{bmatrix} \quad=\quad
# \begin{bmatrix} 6 \\ 10 \\ 6 \\ 17 \end{bmatrix} $$
# 
# &#9989; **<font color=red>DO THIS:</font>**
# Use the functions you wrote above ``vecadd`` and ``sv_multiply`` to write a new function that takes in a list of scalars and a list of vectors (represented by lists) and computes the corresponding linear combination. Call the function ``linear_comb``.

# In[ ]:


# Write your function here:

# def linear_comb(scalars,vectors):
#     return 


# In[ ]:


##Test your function

scalars = [3,-2,1.5]
vectors = [[1,1,0,1],[0,1,0,-1],[2,6,4,8]]

linear_comb(scalars,vectors)


# In[ ]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer.vector(linear_comb([1,-1,2],[[1,1,1],[0,1,1,],[1,0,1]]),'60031b1edacfda4b1f4ebf44ceb05691')


# &#9989; **<font color=red>Bonus Question:</font>**
# Find a collection of scalars that shows that the vector $v = \begin{bmatrix} -2 \\ 3  \\ 2 \end{bmatrix}$ is a linear combination of the vectors $$v_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix}, \quad v_3 =\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}.$$ That is find $a_1,a_2,a_3$ such that $v = a_1 v_1 + a_2 v_2 + a_3 v_3$. 

# In[ ]:


## You can guess and check below, we will discuss how to solve problems like this next week.

a1 = 0
a2 = 0
a3 = 0

scalars = [a1,a2,a3]
vectors = [[1,0,1],[0,1,-1],[0,0,1]]

linear_comb(scalars,vectors)


# In[ ]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer.vector([a1,a2,a3],'be6add17c53320c2147daa4435af47d5')


# ----
# The following video gives an excellent recap and discussion of the vector topics we discussed today. Please watch it.
# 
# * [Direct Link to YouTube Video](https://youtu.be/fNk_zzaMoSs)

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("fNk_zzaMoSs",width=640,height=360, cc_load_policy=True)


# ### Congratulations, we're done!
# 
# If you attend class in-person then have one of the instructors check your notebook and sign you out before leaving class. If you are attending asynchronously, turn in your assignment using D2L.

# Written by Dr. Dirk Colbry and Dr. Matthew Mills, Michigan State University
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# 
# 

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# 102--Matrix_Mechanics_pre-class-assignment.ipynb
# 102-Matrix_Mechanics_in-class-assignment.ipynb
# 103--Linear_Equations_pre-class-assignment.ipynb
# 103-Linear_Equations_in-class-assignment.ipynb
# ```
# 
