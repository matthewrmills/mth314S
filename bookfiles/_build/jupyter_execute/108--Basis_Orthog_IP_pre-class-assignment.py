#!/usr/bin/env python
# coding: utf-8

# 

# ### Readings for Change of Basis
#  * [Heffron Chapter 2 III pg 114-134](http://joshua.smcvt.edu/linearalgebra/book.pdf)
#  * [**_Beezer Subsection CBM pg 549-549_**](http://linear.ups.edu/download/fcla-3.50-tablet.pdf)
#  * [Boyd Section 5.1 pg 91-95](http://vmls-book.stanford.edu/vmls.pdf)
#  
#  ### Readings for Projections
#  * [Heffron Section VI pg 267-275](http://joshua.smcvt.edu/linearalgebra/book.pdf)
#  * [Beezer Subsections OV-GSP pg 154-161](http://linear.ups.edu/download/fcla-3.50-tablet.pdf)
#  * [**_Boyd Section 5.3-5.4 pg 95-102_**](http://vmls-book.stanford.edu/vmls.pdf)
# 
# ### No readings for Inner Products
# 
# 

# # Pre-Class Assignment: Matrix Spaces

# # Goals for today's pre-class assignment 
# 
# </p>
# 
# 1. [Properties of Invertible Matrices](#Properties_of_invertible_Matrices)
# 1. [The Basis of a Vector Space](#The_Basis_of_a_Vector_Space)
# 1. [Change of Basis](#Change_of_Basis)
# 1. [Orthogonal and Orthonormal](#Orthogonal_and_Orthonormal)
# 1. [Gram-Schmidt](#Gram-Schmidt)
# 1. [Inner Products](#Inner_Products)
# 1. [Inner Product on Functions](#Inner_Product_on_Functions)
# 1. [Assignment Wrap-up](#Assignment_Wrap-up)

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import sympy as sym
sym.init_printing(use_unicode=True)


# ----
# <a name="Properties_of_invertible_Matrices"></a>
# 
# # 1.  Review the Properties of Invertible Matrices
# Let $A$ be an $n \times n$ matrix. The following statements are equivalent.
# 
# - The column vectors of $A$ form a basis for $R^n$
# - $|A| \ne 0$
# - $A$ is invertible.
# - $A$ is row equivalent to $I_n$ (i.e. it's reduced row echelon form is $I_n$)
# - The system of equations $Ax = b$ has a unique solution.
# - $rank(A) = n$
# 

# Consider the following example. We claim that the following set of vectors form a basis for $R^3$:
# 
# $$B = \{(2,1, 4), (-1,6, 0), (2, 4, -3) \}$$
# 
# Remember for these two vectors to be a basis they need to obay the following two properties:
# 
# 1. They must span $R^3$. 
# 2. They must be linearly independent.
# 
# Using the above statements we can show this is true in multiple ways.  

# 
# ### The column vectors of $A$ form a basis for $R^n$
# 
# &#9989; **<font color=red>DO THIS:</font>** Define a numpy matrix ```A``` consisting of the vectors $B$ as columns:

# In[2]:


#Put your answer to the above question here


# In[3]:


from answercheck import checkanswer

checkanswer.matrix(A,'94827a40ec59c7d767afe6841e1723ce');


# ### $|A| \ne 0$
# 
# 
# &#9989; **<font color=red>DO THIS:</font>** The first in the above properties tell us that if the vectors in $B$ are truly a basis of $R^3$ then $|A|=0$. Calculate the determinant of $A$ and store the value in ```det```.

# In[ ]:


#Put your answer to the above question here


# In[ ]:


#Verify that the determinate is in fact zero
if np.isclose(det,0):
    print("Since the Determinant is zero the column vectors do NOT form a Basis")
else:
    print("Since the Determinant is non-zero then the column vectors form a Basis.")


# ###  $A$ is invertible.
# 
# 
# &#9989; **<font color=red>DO THIS:</font>** Since the determinant is non-zero we know that there is an inverse to A.  Use python to calculate that inverse and store it in a matrix called ```A_inv```

# In[ ]:


#put your answer to the above question here


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(A_inv,'001aaddd4824f42ad9d2ccde21cf9d24');


# ### $A$ is row equivalent to $I_n$ (i.e. it's reduced row echelon form is $I_n$)
# 

# &#9989; **<font color=red>DO THIS:</font>** According to the property above the reduced row echelon form of an invertable matrix is the Identiy matrix.  Verify using the python ```sympy``` library and store the reduced row echelone matrix in a variable called ```rref``` if you really need to check it.

# In[ ]:


#put your answer to the above question here


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(rref,'cde432847c1c4b6d17cd7bfacc457ed1');


# ## The system of equations $Ax = b$ has a unique solution.
# 
# Let us assume some arbitrary vector $b \in R^n$.  According to the above properties it should only have one solution.
# 
# &#9989; **<font color=red>DO THIS:</font>** Find the solution to $Ax=b$ for the vector $b=(-10,200,3)$. Store the solution in a variable called ```x```

# In[ ]:


from answercheck import checkanswer

checkanswer.vector(x,'161cfd16545b1b5fb13e35d2800f13df');


# ### $rank(A) = n$
# 
# The final property says that the rank should equal the dimension of $R^n$. In our example $n=3$.  Find a ```python``` function to calculate the rank of $A$. Store the value in a variable named ```rank``` to check your answer.
# 

# In[ ]:


#Put your answer to the above quesion here


# In[ ]:


#Verify that the determinate is in fact zero
if np.isclose(rank,3):
    print("Rank is 3")
else:
    print("Rank is not 3. Did we do something wrong?")


# &#9989; **<font color=red>QUESTION (assignment-specific):</font>** Without doing any calculations (i.e. only using the above properties), how many solutions are there to $Ax=0$?  What is(are) the solution(s)?

# Put your answer to the above question here.

# ----
# <a name="The_Basis_of_a_Vector_Space"></a>
# # 2. The Basis of a Vector Space
# 
# Let $U$ be a vector space with basis $B=\{u_1, \ldots, u_n\}$, and let $u$ be a vector in $U$. 
# Because a basis "spans" the vector space, we know that there exists scalars $a_1, \dots, a_n$ such that:
# 
# $$ u = a_1u_1 + \dots + a_nu_n$$
# 
# Since a basis is a linearly independent set of vectors we know the scalars $a_1, \dots, a_n$ are unique.
# 
# The values $a_1, \dots, a_n$ are called the **coordinates of $u$** relative to the basis ($B$) and is typically written as a column vector:
# 
# $$ u_B = 
# \left[
# \begin{matrix}
#     a_1  \\
#     \vdots  \\
#     a_n 
#  \end{matrix}
# \right] 
# $$
# 
# We can create a *transition matrix* $P$ using the inverse of the matrix with the basis vectors being columns. 
# 
# $$P = [ u_1  \ldots  u_n ]^{-1}$$
# 
# Now we will show that matrix $P$ will transition vector $u$ in the standard coordinate system to the coordinates relative to the basis $B$:
# 
# $$ u_B = Pu$$

# **EXAMPLE**: Consider the vector $u = \left[ \begin{matrix} 5 \\ 3 \end{matrix} \right]$ and the basis vectors $B = \{(1,2), (3,-1)\}$. 
# The following code calculates the $P$ *transition matrix* from $B$ and then uses $P$ to calculate the values of $u_B$ ($a_1$ and $a_2$):
# 

# In[ ]:


u = np.matrix([[5],[3]])
sym.Matrix(u)


# In[ ]:


B = np.matrix([[1,2],[3,-1]]).T
sym.Matrix(B)


# In[ ]:


P = np.linalg.inv(B)
ub = P*u

sym.Matrix(ub)


# Here we would like to view this from $R^n$. 
# Let $$B=[u_1 \dots u_n],$$
# then the values of $u_B$ can be found by solving the linear system $$u = B u_B.$$
# The columns of $B$ are a basis, therefore, the matrix $B$ is a $n\times n$ square matrix and it has an inverse. 
# Therefore, we can solve the linear system and obtain 
# $$u_B = B^{-1} u = Pu.$$
# 

# Let's try to visualize this with a plot:

# In[ ]:


ax = plt.axes();


#Blue arrow representing first Basis Vector
ax.arrow(0, 0, B[0,0],B[1,0], head_width=.2, head_length=.2, fc='blue', ec='blue');


#Green arrow representing Second Basis Vector
plt.plot([0,B[0,1]],[0,B[1,1]],color='green'); #Need this line to make the figure work. Not sure why.
ax.arrow(0, 0, B[0,1],B[1,1], head_width=.2, head_length=.2, fc='green', ec='green');

#Original point u as a red dot
ax.scatter(u[0,0],u[1,0], color='red');

plt.show()
#plt.axis('equal');


# Notice that the blue arrow represents the first basis vector and the green arrow is the second basis vector in $B$. 
# The solution to $u_B$ shows 2 units along the blue vector and 1 units along the green vector, which puts us at the point (5,3). 
# 
# This is also called a change in coordinate systems.

# &#9989; **<font color=red>QUESTION</font>**: What is the coordinate vector of $u$ relative to the given basis $B$ in $R^3$?
# 
# $$u = (9,-3,21)$$
# $$B = \{(2,0,-1), (0,1,3), (1,1,1)\}$$
# 
# Store this coordinate in a variable ```ub``` for checking:

# In[ ]:


#Put your answer here


# In[ ]:


from answercheck import checkanswer

checkanswer.vector(ub,'f72f62c739096030e0074c4e1dfca159');


# **_Let's look more closely into the matrix $P$, what is the meaning of the columns of the matrix $P$?_**
# 
# We know that $P$ is the inverse of $B$, therefore, we have $$BP=I.$$
# Then we can look at the first column of the $P$, say $p_{1}$, we have that $Bp_1$ is the column vector $(1,0,0)^\top$, which  is exactly the first component from the standard basis. 
# This is true for other columns. 
# 
# It means that if we want to change an old basis $B$ to a new basis $B'$, we need to find out all the coordinates in the new basis for the old basis, and the transition matrix is by putting all the coordinates as columns.
# 
# Here is the matrix $B$ again:

# In[ ]:


B = np.matrix([[2,0,-1],[0,1,3],[1,1,1]]).T
sym.Matrix(B)


# The first column of P should be the solution to $Bx=\left[ \begin{matrix} 1 \\ 0 \\ 0 \end{matrix} \right]$.  We can use the ```numpy.linalg.solve``` function to find this solution:

# In[ ]:


# The first column of P should be 
u1 = np.matrix([1,0,0]).T
p1 = np.linalg.solve(B,u1)
p1


# We can find a similar answer for columns $p_2$ and $p_3$:

# In[ ]:


# The second column of P should be 
u2 = np.matrix([0,1,0]).T
p2 = np.linalg.solve(B,u2)
p2


# In[ ]:


# The third column of P should be 
u3 = np.matrix([0,0,1]).T
p3 = np.linalg.solve(B,u3)
p3


# In[ ]:


# concatenate three column together into a 3x3 matrix
P = np.concatenate((p1, p2, p3), axis=1)
sym.Matrix(P)


# In[ ]:


# Find the new coordinate in the new basis
u = np.matrix([9,-3,21]).T
UB = P*u
print(UB)


# This should be basically the same answer as you got above. 

# ----
# <a name="Change_of_Basis"></a>
# 
# # 3. Change of Basis
# 
# Now consider the following two bases in $R^2$:
# 
# $$B_1 = \{(1,2), (3,-1)\}$$
# $$B_2 = \{(3,1), (5,2)\}$$
# 
# The transformation from the "standard basis" to $B_1$ and $B_2$ can be defined as the column vectors $P_1$ and $P_2$ as follows:
# 

# In[ ]:


B1 = np.matrix([[1,2],[3,-1]]).T
P1 = np.linalg.inv(B1)

sym.Matrix(P1)


# In[ ]:


B2 = np.matrix([[3,1],[5,2]]).T
P2 = np.linalg.inv(B2)

sym.Matrix(P2)


# &#9989; **<font color=red>DO THIS</font>**: Find the transition matrix $T$ that will take points in the $B_1$ coordinate representation and put them into $B_2$ coordinates.  **_NOTE_** this is analogous to the robot kinematics problem.  We want to represent points in a different coordinate system.

# In[ ]:


# Put your answer to the above question here.


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(T,'dcc03ddff982e29eea6dd52ec9088986')


# &#9989; **<font color=red>QUESTION</font>**: Given $u_{B_1} = \left[ \begin{matrix} 2 \\ 1 \end{matrix} \right]$ (a point named $u$ in the $B_1$ coordinate system) and your calculated transition matrix $T$, what is the same point expressed in the $B_2$ basis (i.e. what is $u_{B2}$)? Store your answer in a variable named ub2 for checking.

# In[ ]:


ub1 = np.matrix([[2],[1]])
sym.Matrix(ub1)


# In[ ]:


##Put your code here


# In[ ]:


from answercheck import checkanswer

checkanswer.vector(ub2,'9a5fe29254c07cf59ebdffcaba679917')


# In general if $\{v_1,\dots,v_n\}$ and $\{u_1,\dots,u_n\}$ are two bases for $\mathbb R^n$, and we define two matrices $B=[v_1 \cdots v_n]$ and C=$[u_1 \cdots u_n]$. If $v$ be a coordinate vector relative to $B$ and $u$ a coordinate vector relative to $C$. The vectors $v$ and $u$ are the same point in $\mathbb R^n$ if and only if  $$Bv=Cu.$$

# ----
# <a name="Orthogonal_and_Orthonormal"></a>
# # 1. Orthogonal and Orthonormal
# 
# When we look at bases like $B = \{(1,2), (3,1)\}$, we see that these two vectors both point up and to the right. It is computationally much better if our basis vectors are always perpendicular to each other. To capture this ideal we define an orthogonal basis below. The normal aspect is the same as earlier in the semester and is just a rescaling so that the dot product of the vector with itself is 1. 
# 
# **Definition:** A set of vectors is said to be **orthogonal** if every pair of vectors in the set is orthogonal (the dot product is 0). 
# The set is **orthonormal** if it is orthogonal and each vector is a unit vector (norm equals 1). 
# 
# **Result:** An orthogonal set of nonzero vectors is linearly independent.
# 
# **Definition:** A basis that is an orthogonal set is called an orthogonal basis.
# A basis that is an orthonormal set is called an orthonormal basis.
# 
# **Result:** Let $\{u_1,\dots,u_n\}$ be an orthonormal basis for a vector space $V$. 
# Then for any vector $v$ in $V$, we have 
# $$v=(v\cdot u_1)u_1+(v\cdot u_2)u_2 +\dots + (v\cdot u_n)u_n$$
# 
# **Definition:** A *square* matrix is **orthogonal** is $A^{-1}=A^\top$.
# 
# **Result:** Let $A$ be a square matrix. The following five statements are equivalent.
# 
# 1. $A$ is orthogonal. 
# 1. The column vectors of $A$ form an orthonormal set. 
# 1. The row vectors of $A$ form an orthonormal set.
# 1. $A^{-1}$ is orthogonal. 
# 1. $A^\top$ is orthogonal.
# 
# **Result:** If $A$ is an orthogonal matrix, then we have $|A|=\pm 1$.
#     
# Consider the following vectors $u_1, u_2$, and $u_3$ that form a basis for $R^3$. 
# 
# $$ u_1 = (1,0,0)$$
# $$ u_2 = (0, \frac{1}{\sqrt(2)}, \frac{1}{\sqrt(2)})$$
# $$ u_3 = (0, \frac{1}{\sqrt(2)}, -\frac{1}{\sqrt(2)})$$

# &#9989; **<font color=red>DO THIS:</font>**  Show that the vectors $u_1$, $u_2$, and $u_3$ are linearly independent:

# Put your answer to the above here

# &#9989; **<font color=red>QUESTION 1:</font>** How do you show that $u_1$, $u_2$, and $u_3$ are orthogonal?

# Put your answer to the above question here

# &#9989; **<font color=red>QUESTION 2:</font>** How do you show that $u_1$, $u_2$, and $u_3$ are normal vectors?

# Put your answer to the above question here

# &#9989; **<font color=red>DO THIS:</font>**  Express the vector $v = (7,5,-1)$ as a linear combination of the $u_1$, $u_2$, and $u_3$ basis vectors:

# In[ ]:


# Put your answer here


# ----
# <a name="Gram-Schmidt"></a>
# # 5. Gram-Schmidt
# 
# 
# Watch this video for the indroduction of the Gram-Schmidt process, which we will implement in class. It is an algorithm that takes a set of basis vectors and produces a set of orthogonal basis vectors.

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("rHonltF77zI",width=640,height=360, cc_load_policy=True)


# ---
# <a name="Inner_Products"></a>
# # 6. Inner Products
# 
# Notice how above the process of checking orthogonal vectors, and even the normilization required the use of the dot product. There is however a generalization of dot products that gives other ways to measure distance. 
# 
# **Definition:** An **inner product** on a vector space $V$ (Remember that $R^n$ is just one class of vector spaces) is a function that takes as input two vectors and produces a number.
# We typically denote the inner product of vectors $u$ and $v$ by the notation $\langle u,v \rangle$. This function satisfies the following conditions for vectors $u, v, w$ and scalar $c$:
# 
# - $\langle u,v \rangle = \langle v,u \rangle$ (symmetry axiom)
# - $\langle u+v,w \rangle = \langle u,w \rangle + \langle v,w \rangle$ (additive axiom) 
# - $\langle cu,v \rangle = c\langle v,u \rangle$ (homogeneity axiom)
# - $\langle u,u \rangle \ge 0 \text{ and } \langle u,u \rangle = 0 \text{ if and only if } u = 0$ (positive definite axiom) 
# 
# The inner product by the formula $u \cdot v = u_1 v_1 + \dots + u_n v_n$ is an example of an inner product.
# 
# However we will see other formulas that give inner products in the future. One example for a different inner product on $R^2$ is 
# 
# $$\langle u,v \rangle = 2u_1v_1 + 3u_2v_2.$$
# 
# Notice now that if $u = [1,2]$ and $v =[-3,1]$ we have that $u \cdot v = -1$, but $\langle u,v \rangle = 2(1)(-3)+3(2)(1) = 0.$
# 
# Regardless of what your formula is for your inner product everything related to distance and orthogonality can be computed with it. The formulas are given below.
# 
# 
# ## Norm of a vector
# 
# **Definition:** Let $V$ be an inner product space. The **norm** of a vector $v$ is denoted by $\| v \|$ and is defined by:
# 
# $$\| v \| = \sqrt{\langle v,v \rangle}.$$
# 
# ## Angle between two vectors
# 
# **Definition:** Let $V$ be a real inner product space. The **angle $\theta$ between two nonzero vectors $u$ and $v$** in $V$ is given by:
# 
# $$cos(\theta) = \frac{\langle u,v \rangle}{\| u \| \| v \|}.$$
# 
# ## Orthogonal vectors
# 
# **Definition:** Let $V$ be an inner product space.  Two vectors $u$ and $v$ in $V$ are **orthogonal** if their inner product is zero:
# 
# $$\langle u,v \rangle = 0.$$
# 
# ## Distance
# **Definition:** Let $V$ be an inner product space. The **distance between two vectors (points) $u$ and $v$** in $V$ is denoted by $d(u,v)$ and is defined by:
# 
# $$d(u,v) = \| u-v \| = \sqrt{\langle u-v, u-v \rangle}$$
# 
# 
# ### Example:
# Let $R^2$ have an inner product defined by:
# $$\langle (a_1,a_2),(b_1,b_2)\rangle = 2a_1b_1 + 3a_2b_2.$$

# &#9989; **<font color=red>QUESTION 1:</font>** What is the norm of (1,-2) in this space?

# Put your answer to the above question here.

# &#9989; **<font color=red>QUESTION 2:</font>** What is the distance between (1,-2) and (3,2) in this space?

# Put your answer to the above question here.

# &#9989; **<font color=red>QUESTION 3:</font>** What is the angle between (1,-2) and (3,2) in this space?

# Put your answer to the above question here.

# &#9989; **<font color=red>QUESTION 4:  (assignment-specific)</font>** Determine if (1,-2) and (3,2) are orthogonal in this space?

# Put your answer to the above question here.

# ---
# <a name="Inner_Product_on_Functions"></a>
# # 2. Inner Product on Functions
# 
# Recall that we said that collections of polynomials can also form vector spaces. In this system we can introduce a formula for an inner product using integrals. 

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("8ZyeHtgMBjk",width=640,height=360, cc_load_policy=True)


# ### Example
# Consider the following two functions as elements of the vector space of polynomials
# 
# $$f(x)=3x-1$$
# $$g(x)=5x+3$$
# 
# $$\text{with inner product defined by }\langle f,g\rangle=\int_0^1{f(x)g(x)dx}.$$
# 
# &#9989; **<font color=red>QUESTION 5:</font>** What is the norm of $f(x)$ in this space?

# Put your answer to the above question here. (Hint: you can use `sympy.integrate` to compute the integral)

# &#9989; **<font color=red>QUESTION 6:</font>** What is the norm of g(x) in this space?

# Put your answer to the above question here.

# &#9989; **<font color=red>QUESTION 7:</font>** What is the inner product of $f(x)$ and $g(x)$ in this space?

# Put your answer to the above question here.

# ---
# <a name=Assignment-wrap-up></a>
# ## 6. Assignment wrap up
# 
# 
# Please fill out the form that appears when you run the code below.  **You must completely fill this out in order to receive credit for the assignment!** If you cannont load the form below please try logging in to [spartan365.msu.edu](http://spartan365.msu.edu/) and try running it again, or simply use the direct link provided below. 
# 
# [Direct Link to Microsoft Form](https://forms.office.com/r/n0PEF9xt59)
# 

# &#9989; **<font color=red>QUESTION (assignment-specific):</font>** Without doing any calculations (i.e. only using the above properties), how many solutions are there to $Ax=0$?  What is(are) the solution(s)?

# &#9989; **<font color=red>QUESTION 4  (assignment-specific):</font>** Determine if (1,-2) and (3,2) are orthogonal in this space?

# In[ ]:


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

# ----

# Written by Dr. Dirk Colbry, Michigan State University
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# ----

# 
# 
