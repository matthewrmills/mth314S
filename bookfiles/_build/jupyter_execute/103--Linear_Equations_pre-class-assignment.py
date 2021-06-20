#!/usr/bin/env python
# coding: utf-8

# 

# # 103 Pre-Class Assignment: Linear Equations

# Choose one reading out  each of the following groups
# 
# ### Readings for Systems of Equations (Recommended in bold)
#  * [Heffron Chapter 1.I, pg 1-2](http://joshua.smcvt.edu/linearalgebra/book.pdf)
#  * [Beezer Chapter SLE pg 1-7](http://linear.ups.edu/download/fcla-3.50-tablet.pdf)
# 
# 
# ### Recommended further readings for Elementary Operations & Gauss Jordan
# 
# * **_[Beezer - Section RREF pg 22-44](http://linear.ups.edu/download/fcla-3.50-tablet.pdf)_**
# * [Heffron - Chapter 1.I, pg 2-13](http://joshua.smcvt.edu/linearalgebra/book.pdf)
# 
# 

# ---
# ### Assignment Overview
# 1. [System of Linear Equations](#System_of_Linear_Equations)
# 1. [Visualizing the problem](#Visualizing_the_problem)
# 2. [Introduction to Gauss Jordan Elimination](#Introduction-to-Gauss-Jordan-Elimination)
# 3. [ Gauss Jordan Elimination and the Row Echelon Form](#-Gauss-Jordan-Elimination-and-the-Row-Echelon-Form)
# 4. [Gauss Jordan Practice](#Gauss-Jordan-Practice)
# 5. [Assignment wrap up](#Assignment-wrap-up)
# 
# 

# ---
# <a name="System_of_Linear_Equations"></a>
# ## 1. System of Linear Equations
# 
# In this course we will spend a lot of time working with systems of linear equations.  A linear equation is in the form:
# 
# $$a_1x_1 + a_2x_2 + a_3x_3 + \ldots + a_nx_n = b$$
# 
# Where $a_1, a_2, a_3, \ldots a_n$ and $b$ are known constants and $x_1, x_2, x_3, \ldots x_n$ are unknown values.  Typically we have systems of equations with different values of $a$s and $b$s but the unknowns are the same.  
# 
# The following video explains the different ways we can describe linear systems.

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo("AQJeOg4ZoIk",width=640,height=360, cc_load_policy=True)


# The following is a summary of the syntax shown in the video:

# ### Linear Equation $$b = a_1x_1+a_2x_2+a_3x_3 + \ldots a_nx_n$$

# ### System of linear equations
# $$b_1 = a_{11}x_1+a_{12}x_2+a_{13}x_3 + \ldots a_{1n}$$
# $$b_2 = a_{21}x_1+a_{22}x_2+a_{23}x_3 + \ldots a_{2n}$$
# $$b_3 = a_{31}x_1+a_{32}x_2+a_{33}x_3 + \ldots a_{3n}$$
# $$\vdots$$
# $$b_m = a_{m1}x_1+a_{m2}x_2+a_{m3}x_3 + \ldots a_{mn}$$

# ### System of linear equations (Matrix format)

# $$ 
# \left[ 
# \begin{matrix}
#     b_1 \\ 
#     b_2 \\
#     b_3 \\
#     \vdots \\
#     b_m
#  \end{matrix}
# \right] 
# =
# \left[ 
# \begin{matrix}
#  a_{11} & a_{12} & a_{13} &   & a_{1n} \\ 
#  a_{21} & a_{22} & a_{23} &  \ldots & a_{2n} \\ 
#   a_{31} & a_{32} & a_{33} &   & a_{3n} \\ 
#   & \vdots &   & \ddots & \vdots \\ 
#  a_{m1} & a_{m2} & a_{m3} &   & a_{mn} 
# \end{matrix}
# \right] 
# \left[ 
# \begin{matrix}
#     x_1 \\ 
#     x_2 \\
#     x_3 \\
#     \vdots \\
#     x_m
# \end{matrix}
# \right] $$
# 
# $$b=Ax$$

# ### System of linear equations (Augmented Form)
# $$ 
# \left[ 
# \begin{matrix}
#  a_{11} & a_{12} & a_{13} &   & a_{1n} \\ 
#  a_{21} & a_{22} & a_{23} &  \ldots & a_{2n} \\ 
#   a_{31} & a_{32} & a_{33} &   & a_{3n} \\ 
#   & \vdots &   & \ddots & \vdots \\ 
#  a_{m1} & a_{m2} & a_{m3} &   & a_{mn} 
# \end{matrix}
#  \, \middle\vert \,
# \begin{matrix}
#     b_1 \\ 
#     b_2 \\
#     b_3 \\
#     \vdots \\
#     b_m
# \end{matrix}
# \right] 
# $$

# Consider the example of linear equations in the following video. 
# 
# &#9989; **<font color=red>TODO:</font>**  Watch the video and follow along in the notebook.

# In[2]:


from IPython.display import YouTubeVideo
YouTubeVideo("CH68cc7sH4A",width=640,height=360, cc_load_policy=True)


# 
# > Giselle works as a carpenter and as a blacksmith. She earns 20 dollars per hour as a carpenter and 25 dollars 
# > per hour as a blacksmith. Last week, Giselle worked both jobs for a total of 30 hours, and earned a total of 690 dollars. How long did Giselle work as a carpenter last week, and how long did she work as a blacksmith?
# >
# > - [Brainly.com](https://brainly.com/question/2202719)
# 
# This problems gives us two equations and two unknowns:
# 
# $$ c + b = 30 $$
# $$ 20c + 25b = 690 $$
# 
# <a name="solution"></a>
# How would we solve this in linear algebra?  
# 
# $$ c + b = 30$$
# $$ 20c + 25b = 690$$
# 
# First, we can multiply the first equation by -20 and add to the second equation.  This is often called a "linear combination" of the two equations. The operation does not change the answer:
# 
# $$ -20c - 20b = -600$$
# $$ 20c + 25b = 690$$
# $$----$$
# $$ 0c + 5b = 90$$
# 
# This is our new system of equations:
# $$ c + b = 30$$
# $$ 0c + 5b = 90$$
# 
# Now we can easily divide the second equation by 5 and get the value for $b$:
# 
# $$b = 90/5 = 18$$
# 
# If we substitute 18 for $b$ into the first equation we get:
# $$ c + 18 = 30$$
# 
# And solving for $c$ gives us $c = 30-18=12$.  Let's check to see if this works by substituting $b=18$ and $c=12$ into our original equations:
# 
# $$ 12 + 18 = 30$$
# $$ 20(12) + 25(18) = 690$$

# Let's check the answer using Python:

# In[3]:


b = 18
c = 12


# In[4]:


c + b == 30


# In[5]:


20*c + 25*b == 690


# &#9989; **<font color=red>QUESTION:</font>**  The above video described three (3) elementary operators that can be applied to a system of linear equations and not change their answer.  What are these three operators?  

# **_Erase the contents of this cell and put your answer to the above question here_**

# ---
# <a name="Visualizing_the_problem"></a>
# ## 2.  Visualizing the problem
# We can visualize the solution to a system of linear equations in a graph. If we make $b$ the "$y$"-axis and $c$ the "$x$"-axis. For each equation, we calculate the $b$ value for each $c$, and two equations give us two lines. 
# 
# **Note:** This is sometimes called the "Row Picture." I will ask you why it has this name in class so think about it.
# 

# In[6]:


from IPython.display import YouTubeVideo
YouTubeVideo("BSxWO6FGib0",width=640,height=360, cc_load_policy=True)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np


# In[8]:


c = np.linspace(0,20)
c


# In[9]:


b1 = 30-c
b2 = (690-20*c)/25


# ### Row Picture

# In[10]:


plt.plot(c,b1)
plt.plot(c,b2)
plt.xlabel('c (hours worked as carpenter)')
plt.ylabel('b (hours worked as blacksmith)')
plt.scatter(12,18);


# 
# Now, consider the next set of equations which do not have a solution
# 
# $$-2x+y=3$$
# $$-4x+2y=2$$
# 

# In[11]:


x = np.linspace(-10,10)
y1 =  3+2*x
y2 = (2+4*x)/2
plt.plot(x,y1)
plt.plot(x,y2);


# In[12]:


from IPython.display import YouTubeVideo
YouTubeVideo("Z9gkovHDpIQ",width=640,height=360, cc_load_policy=True)


# Consider the next set of equations which have infinite many solutions
# 
# $$4x-2y=6$$
# $$6x-3y=9$$
# 

# In[13]:


x = np.linspace(-10,10)
y1 =  (4*x-6)/2
y2 = (6*x-9)/3
plt.plot(x,y1)
plt.plot(x,y2)


# &#9989; **<font color=red>DO THIS:</font>**  Plot the following equations from -100 to 100
# 
# $$ 18x+21y = 226$$
# $$ 72x-3y = 644$$

# In[14]:


# Put your python code here


# &#9989; **<font color=red>QUESTION:</font>**  Using the graph, what is a visual estimation of the solution to these two equations?  Hint, you may want to change the $x$ range to "zoom" in on the intersection. 

# **_Put your answer to the above question here._**

# ### Column Picture
# 
# I think a good programmer is a lazy person. Let's avoid writing all of the letters in the above equation by changing it into a column vector format as follows.
# 

# $$ 
# c
# \left[
# \begin{matrix}
#     1 \\ 20  
#  \end{matrix}
# \right]
# +
# b
# \left[
# \begin{matrix}
#     1 \\ 5  
#  \end{matrix}
# \right]
# =
# \left[
# \begin{matrix}
#  30 \\ 330
# \end{matrix}
# \right]
# $$

# Notice that this still represents the same system of equations. We just write the constants as column vectors and we only have to write the unknowns once (Since they are the same for all equations). 

# Let's plot this "column picture", which shows how the above equation is a "linear combination" of the two column vectors.  
# 
# One way to think about this is we can only move in straight lines in two directions. The first direction is (1,20) and the second is (1,5).  The solution to the problem is how far in each direction we need to move to arrive at our final destination of (30,330).
# 
# The first column is a vector in the (1,20) direction. The variable $c$ is how far in the (1,20) direction we want to go.  Then $b$ is how far in the (1,5) direction we want to go to arrive at the point (30,330).
# 
# We will use the ```matplotlib``` function ```arrow``` to plot the vectors.  The arrow function takes a starting point $[x,y]$ and a direction $[dx,dy]$ as inputs and draws an arrow from the starting point in the direction specified.

# First thing to do is plot the first column as a vector. From the origin (0,0) to $c\left[
# \begin{matrix}
#     1 \\ 20  
#  \end{matrix}
# \right]$
# 
# **or** $x = c$ and $y = 20c$ with $c=12$

# In[15]:


c = 12

#hack to initialize bounds of plot (need this to get the arrows to work?)
plt.plot(0,0)
plt.plot(30,330)

# Plot the first arrow 
plt.arrow(0, 0, c*1, c*20,head_width=2, head_length=10, fc='blue', ec='blue')


# Next thing to do is plot the second column as a vector by adding it to the first. This ```arrow``` will start at the end of the previous vector and "add" the second column vector:

# In[16]:


b = 18

#hack to inicialize bounds of plot (need this to get the arrows to work?)
plt.plot(0,0)
plt.plot(30,330)

# Plot the first arrow
plt.arrow(0, 0, c*1, c*20,head_width=2, head_length=10, fc='blue', ec='blue')

#Plot the second arrow starting at the end of the first
plt.arrow(c, c*20, b*1, b*5, head_width=2, head_length=10, fc='red', ec='red')


# The takeaway to this figure is that these two column vectors, when added together, end up at the point that represents the right hand side of the above equation (i.e. (30, 330)). 

# In[17]:


#hack to inicialize bounds of plot (need this to get the arrows to work?)
plt.plot(0,0)
plt.plot(30,330)

# Plot the first arrow
plt.arrow(0, 0, c*1, c*20,head_width=2, head_length=10, fc='blue', ec='blue')

#Plot the second arrow starting at the end of the first
plt.arrow(c, c*20, b*1, b*5, head_width=2, head_length=10, fc='red', ec='red')

#Plot a righthand column vector as a point.
plt.arrow(0,0, 30, 330, head_width=2, head_length=10, fc='purple', ec='purple')
plt.xlabel('x');
plt.ylabel('y');


# We say that two column vectors "**span**" the $xy$-plane if any point on the x,y plane can be represented as a linear combination of the two vectors. 
# 
# For example the vectors $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$ span the $xy$-plane because any point $\begin{bmatrix} a \\ b \end{bmatrix}$ can be written as the linear combination $$\begin{bmatrix} a \\ b \end{bmatrix} = a \begin{bmatrix} 1 \\ 0 \end{bmatrix} + b \begin{bmatrix} 0 \\ 1 \end{bmatrix}.$$
# 
# &#9989; **<font color=red> QUESTION:</font>** Give an example of two column vectors that do **NOT** span the $xy-$plane:

# **_Put your answer to the above question here._**

# ---
# <a name=Introduction-to-Gauss-Jordan-Elimination></a>
# ## 3. Introduction to Gauss Jordan Elimination
# 
# As we mentioned above a linear system of equations can be written in an "Augmented matrix" format. The example about Giselle is represented by 
# 
# $$ 
# \left[
# \begin{matrix}
#     1 & 1 \\ 
#     20 & 25
#   \end{matrix}
# \, \middle\vert \,
# \begin{matrix}
#  30 \\ 
#  690
# \end{matrix}
# \right] 
# $$
# 
# The [solution provided above](#solution) makes use of the elementary row operations 
# 
# 1. Interchange two rows of a matrix
# 2. Multiply the elements of a row by a nonzero constant
# 3. Add a multiple of the elements of one row to the corresponding elements of another
# 
# **(Notice these are the same things as what the elementary matrix multiplication does.)**
# 
# This is just one example of a method called Gauss-Jordan elimination, which we will thouroughly discuss in class tomorrow. For now we leave you with just a brief introduction. 

# In[18]:


from IPython.display import YouTubeVideo
YouTubeVideo("iGmtmF_hm2g",width=640,height=360, cc_load_policy=True)


# Consider the element $a_{2,1}$ in the following $A$ Matrix.  
# $$ 
# A = \left[
# \begin{matrix}
#     1 & 1 \\ 
#     20 & 25  
#  \end{matrix}
#  \, \middle\vert \,
# \begin{matrix}
#  30 \\ 
#  690
# \end{matrix}
# \right] 
# $$

# &#9989; **<font color=red>ASSIGNMENT SPECIFIC QUESTION:</font>** : Describe an elementary row operation that could be used to make element $a_{(2,1)}$ zero?  

# Put your answer here. 

# &#9989; **<font color=red>QUESTION:</font>** : What is the new matrix given the above row operation.  

# Modify the contents of this cell and put your answer to the above question here.  
# $$ 
# A = \left[
# \begin{matrix}
#     1 & 1 \\ 
#     0 & ??  
#  \end{matrix}
#  \, \middle\vert \,
# \begin{matrix}
#  30 \\ 
#  ??
# \end{matrix}
# \right] 
# $$
# 
# 
# 
# **Hint**, we are using a formating language called Latex to display the above matrix. You should just be able to replace the ?? with your new numbers. If you can't figure out what is going on, try searching the web with "latex math and matrix." If it still doesn't make sense, format your answer in another way that will be clear to understand by the you and the instructor.

# ---
# As a recap, The general augmented matrix format for a system of linear equations can be written as follows:
# 
# $$ 
# X = 
# \left[
# \begin{matrix}
#     x_{11}       & x_{12} & x_{13} & \dots  \\
#     x_{21}       & x_{22} & x_{23} & \dots \\
#     \ldots       & \ldots & \ldots & \ddots \\
#     x_{m1}       & x_{m2} & x_{m3} & \dots 
#  \end{matrix}
# \, \middle\vert \,
# \begin{matrix}
# x_{1n} \\ x_{2n} \\ \ldots \\ x_{mn}
# \end{matrix}
# \right] ^{mxn}
# $$
# 
# where $x_{ij}$ is a scalar element in the matrx.

# The following system of linear equations:
# 
# $$x_1 = 2.14159$$
# $$x_2 = 4$$
# $$x_3 = -7.2$$
# $$x_4 = 69$$
# $$x_5 = 84$$
# $$x_6 = 240$$
# 
# can be rewritten into the augmented matrix:
# 
# 
# $$ 
# X = 
# \left[
# \begin{matrix}
#     1 & 0 & 0 & 0  & 0 & 0  \\
#     0 & 1 & 0 & 0  & 0 & 0  \\
#     0 & 0 & 1 & 0  & 0 & 0  \\
#     0 & 0 & 0 & 1  & 0 & 0  \\
#     0 & 0 & 0 & 0  & 1 & 0  \\
#     0 & 0 & 0 & 0  & 0 & 1  
#  \end{matrix}
# \, \middle\vert \,
# \begin{matrix}
# 2.14159 \\ 4 \\ -7.2 \\ 69 \\ 84 \\ 240
# \end{matrix}
# \right] ^{6x7}
# $$
# 
# Notice the submatrix on the left hand side is just the $I_6$ identity matrix and the right hand side are the solutions. When a system of linear equations has a unique solution, it is always possible (using Gauss-Jordan elimination) to perform elementary row operations on the augmented matrix to get the identity matrix on the left side.

# 
# <a name=-Gauss-Jordan-Elimination-and-the-Row-Echelon-Form></a>
# ## 4.  Gauss Jordan Elimination and the Row Echelon Form
# 
# 

# In[19]:


from IPython.display import YouTubeVideo
YouTubeVideo("v6RstFsrTJY",width=640,height=360, cc_load_policy=True)


# The above video left out a special case for Reduced Row Echelon form. There can be non-zero elements in columns that do not have a leading one. For example, All of the following are in Reduced Row Echelon form:
# 
# $$ 
# \left[
# \begin{matrix}
#     1 & 2 & 0 & 3 & 0 & 4 \\ 
#     0 & 0 & 1 & 2 & 0 & 7 \\ 
#     0 & 0 & 0 & 0 & 1 & 6 \\ 
#     0 & 0 & 0 & 0 & 0 & 0  
# \end{matrix}
# \right] 
# $$
# 
# 
# $$ 
# \left[
# \begin{matrix}
#     1 & 2 & 0 & 0 & 4 \\ 
#     0 & 0 & 1 & 0 & 6 \\ 
#     0 & 0 & 0 & 1 & 5   
# \end{matrix}
# \right] 
# $$

# &#9989; **<font color=red>QUESTION:</font>** : What are the three steps in the Gauss-Jordan Elimination algorithm?

#  Put your answer here. 

# 
# 
# ---
# <a name=Gauss-Jordan-Practice></a>
# ## 5. Gauss Jordan Practice
# 
# 
# 
# 
# &#9989; **<font color=red>DO THIS:</font>**: Solve the following system of linear equations using the Gauss-Jordan algorithm.  Try to do this before watching the video!
# 
# $$x_1 + x_3 = 3$$
# $$2x_2 - 2x_3 = -4$$
# $$x_2 - 2x_3 = 5$$

# Put your answer here

# In the following video, we solve the same set of linear equations. Watch the video after trying to do this on your own.  It is provided here in case you get stuck.  

# In[20]:


from IPython.display import YouTubeVideo
YouTubeVideo("xT16yIVw_KE",width=640,height=360, cc_load_policy=True)


# &#9989; **<font color=red>QUESTION:</font>**: Something was unclear in the above videos.  Describe the difference between a matrix in "row echelon" form and "reduced row echelon" form. 

# **_Put your answer to the above question here_**

# ---
# <a name=Assignment-wrap-up></a>
# ## 6. Assignment wrap up
# 
# 
# Please fill out the form that appears when you run the code below.  **You must completely fill this out in order to receive credit for the assignment!** If you cannont load the form below please try logging in to [spartan365.msu.edu](http://spartan365.msu.edu/) and try running it again, or simply use the direct link provided below. 
# 
# [Direct Link to Microsoft Form](https://forms.office.com/r/n0PEF9xt59)
# 

# The following question was answered eariler in the notebook. Please copy and paste the question and your answer into the survey.
# 
# &#9989; <font color=red>**Assignment-Specific QUESTION:**</font> Describe an elementary row operation that could be used to make element $a_{(2,1)}$ zero?  

# Put your answer to the above question here and then copy and paste it into the survey.

# In[21]:


from IPython.display import HTML
HTML(
"""
<iframe width="640px" height= "480px" src= "https://forms.office.com/Pages/ResponsePage.aspx?id=MHEXIi9k2UGSEXQjetVofSS1ePbivlBPgYEBiz_zsf1UOTk3QU5VVEo1SVpKWlpaWlU4WTlDUlQwWi4u&embed=true" frameborder= "0" marginwidth= "0" marginheight= "0" style= "border: none; max-width:100%; max-height:100vh" allowfullscreen webkitallowfullscreen mozallowfullscreen msallowfullscreen> </iframe>
"""
)


# ---------
# ### Congratulations, we're done!
# 
# ###EndPreClass###

# ### Course Resources:
# 
# 
# 
# 
# 

# Written by Dr. Dirk Colbry and Dr. Matthew Mills, Michigan State University
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# 
# 
