#!/usr/bin/env python
# coding: utf-8

# 

# # 109 Pre-Class Assignment: Least Squares Fit (Regression) and Review

# ### Readings for this topic (Recommended in bold)
#  * [Heffron Chapter 3 pg 287-292](http://joshua.smcvt.edu/linearalgebra/book.pdf)
#  * [**_Boyd Chapter 13 pg 225-239_**](http://vmls-book.stanford.edu/vmls.pdf)
# 

# 
# # Goals for today's pre-class assignment 
# 
# </p>
# 
# 1. [Least Squares Fit](#Least_Squares_Fit)
# 1. [Linear Regression](#Linear_Regression)
# 1. [One-to-oneand Inverse transform](#One-to-one_and_Inverse_transform)
# 1. [Inverse of a Matrix](#Inverse_of_a_Matrix)
# 1. [Review](#review)
#     1. [Linear Systems](#Linear-Systems)
#     2. [Under Defined Systems](#Under-Defined-Systems)
#     3. [Invertible Systems](#Invertible-Systems)
#     4. [Overdefined systems](#Overdefined-systems)
#     5. [System Properties](#System-Properties)
# 6. [Assignment wrap up](#Assignment-wrap-up)

# ----
# <a name="Least_Squares_Fit"></a>
# # 1. Least Squares Fit
# 
# **Review Chapters Chapter 13 pg 225-239 of the Boyd textbook.**
# 
# In this first part of this course, we try to solve the system of linear equations $Ax=b$ with an $m\times n$ matrix $A$ and a column vector $b$. 
# 
# There are three possible outcomes: an unique solution, no solution, and infinite many solutions. 
# (Review the material on this part if you are no familiar with when the three types of outcomes happen.)
# 
# When $m<n$, we call the matrix $A$ underdeterminated, because we can not have an unique solution for it. 
# When $m>n$, we call the matrix $A$ overdeterminated, becasue we may not have a solution with high probability. 
# 
# However, if we still need to find a best $x$, even when there is no solution or infinite many solutions we use a technique called least squares fit (LSF). Least squares fit find $x$ such that $\|Ax-b\|$ is the smallest (i.e. we try to minimize the estimation error).

# + When there is no solution, we want to find $x$ such that $Ax-b$ is small (here, we want $\|Ax-b\|$ to be small). 
# + If the null space of $A$ is just $\{0\}$, we can find an unique $x$ to obtain the smallest $\|Ax-b\|$.
#     + If there is a unique solution $x^*$ for $Ax=b$, then $x^*$ is the optimal $x$ to obtain the smallest $\|Ax-b\|$, which is 0.
#     + Because the null space of $A$ is just $\{0\}$, you can not have infinite many solutions for $Ax=b$.
# + If the null space of $A$ is not just $\{0\}$, we know that we can always add a nonzero point $x_0$ in the null space of $A$ to a best $x^*$, and $\|A(x^*+x_0)-b\|=\|Ax^*-b\|$. Therefore, when we have multiple best solutions, we choose to find the $x$ in the rowspace of $A$, and this is unique. 

# **<font color=red>QUESTION 1:</font>** Let $$A=\begin{bmatrix}1\\2\end{bmatrix},\quad b=\begin{bmatrix}1.5 \\ 2\end{bmatrix}$$
# Find the best $x$ such that $\|Ax-b\|$ has the smallest value.

# Put your answer to the above question here.

# **<font color=red>QUESTION 2:</font>** Compute $(A^\top A)^{-1}A^\top b$.

# Put your answer to the above question here.

# ----
# <a name="Linear_Regression"></a>
# # 2. Linear Regression
# 
# Watch the video for using Least Squares to do linear regression.

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo("Lx6CfgKVIuE",width=640,height=360, cc_load_policy=True)


# **<font color=red>QUESTION 3:</font>** How to tell it is a good fit or a bad one?

# Put your answer to the above question here.

# ----
# <a name="One-to-one_and_Inverse_transform"></a>
# # 3. One-to-one and Inverse transform
# 
# Read Section 4.9 of the textbook if you are not familiar with this part. 
# 
# **Definition:** A transformation $T:U\mapsto V$ is said to be *one-to-one* if each element in the range is the image of just one element in the domain. That is, for two elements ($x$ and $y$) in $U$. $T(x)=T(y)$ happens only when $x=y$.
# 
# **Theorem:** Let $T:U\mapsto V$ be a one-to-one linear transformation. If $\{u_1,\dots,u_n\}$ is linearly independent in $U$, then $\{T(u_1),\dots,T(u_n)\}$ is linearly independent in $V$. 
# 
# **Definition:** A linear transformation $T:U\mapsto V$ is said to be *invertible* if there exists a transformation $S:V\mapsto U$, such that 
# $$S(T(u))=u,\qquad T(S(v))=v,$$
# for any $v$ in $V$ and any $u$ in $U$.

# **<font color=red>QUESTION 4:</font>** If linear transformation $T:U\mapsto V$ is invertible, and the dimension of $U$ is 2, what is the dimension of $V$? Why?

# Put your answer to the above question here.

# ----
# <a name="Inverse_of_a_Matrix"></a>
# # 4. Inverse of a Matrix
# 
# + Recall the four fundamental subspaces of a $m\times n$ matrix $A$
#     + The rowspace and nullspace of $A$ in $R^n$
#     + The columnspace and the nullspace of $A^\top$ in $R^m$

# + The two-sided inverse gives us the following
# $$ {A}{A}^{-1}=I={A}^{-1}{A} $$
#     + For this we need $r = m = n$, here $r$ is the rank of the matrix.

# + For a left-inverse, we have the following
#     + Full column rank, with $r = n \leq m$ (but possibly more rows)
#     + The nullspace contains just the zero vector (columns are independent)
#     + The rows might not all be independent
#     + We thus have either no or only a single solution to $Ax=b$.
#     + $A^\top $ will now also have full row rank
#     + From $(A^\top A)^{-1}A^\top A = I$ follows the fact that $(A^\top A)^{-1}A^\top$ is a left-sided inverse
#     + Note that $(A^\top A)^{-1}A^\top$ is a $n\times m$ matrix and $A$ is of size $m\times n$, theire mulitiplication $(A^\top A)^{-1}A^\top A$ results in a $n\times n$ identity matrix
#     + The $A(A^\top A)^{-1}A^\top$ is a $m\times m$ matrix. BUT $A(A^\top A)^{-1}A^\top\neq I$ if $m\neq n$. The matrix $A(A^\top A)^{-1}A^\top$ is the projection matrix onto the column space of $A$. 

# **<font color=red>QUESTION 5:</font>** What is the projection matrix that projects any vector onto the subspace spanned by $[1,2]^\top$. (What matrix will give the same result as projecting any point onto the vector $[1,2]^\top$.)

# Put your answer to the above question here.

# **<font color=red>QUESTION 6:</font>** If $m=n$, is the left inverse the same as the inverse?

# Put your answer to the above question here.

# **Theorem:** For a matrix $A$ with $r=n<m$, the columnspace of $A$ has dimension $r(=n)$. The linear transfrom $A: R^n\mapsto R^m$ is one-to-one. In addition, the linear transformation $A$ from $R^n$ to the columnspace of $A$ is one-to-one and onto (it means that for any element in the columnspace of $A$, we can find $x$ in $R^n$ such that it equals $Ax$.) 
# Then the left inverse of $A$ is a one-to-one mapping from the columnspace of $A$ to $R^n$, and it can be considered as an inverse transform of $A$. 

# ----
# <a name=review></a>
# # 5. Review
# Everything below here is a review.

# <a name=Linear-Systems></a>
# ## A. Linear Systems
# 
# In this course, we learned how to represent linear systems which basically consists of equations added sums of multiple numbers in the form:
# 
# $$b = a_1x_1+a_2x_2+a_3x_3 + \ldots a_mx_m$$
# 
# Systems of linear equations are multiple equations of the above form with basically the same unknowns but different values of $a$ and $b$. 
# 
# $$b_1 = a_{11}x_1+a_{12}x_2+a_{13}x_3 + \ldots a_{1n}x_n$$
# $$b_2 = a_{21}x_1+a_{22}x_2+a_{23}x_3 + \ldots a_{2n}x_n$$
# $$b_3 = a_{31}x_1+a_{32}x_2+a_{33}x_3 + \ldots a_{3n}x_n$$
# $$\vdots$$
# $$b_m = a_{m1}x_1+a_{m2}x_2+a_{m3}x_3 + \ldots a_{mn}x_n$$
# 
# The above equations can be represented in matrix form as follows:
# 
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
#     x_n
# \end{matrix}
# \right] 
# $$
# 
# Which can also be represented in "augmented" form as follows:
# 
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

# The above systems can be modified into equivelent systems using combinations of the following operators. 
# 
# 1. Multiply any row of a matrix by a constant
# 2. Add the contents of one row by another row.
# 3. Swap any two rows. 
# 
# Often the 1st and 2nd operator can be combined where a row is multipled by a constanet and then added (or subtracted) from another row. 

# &#9989; **<font color=red>QUESTION:</font>**  Consider the matrix $A= \left[ 
# \begin{matrix} 1 & 3 \\ 0 & 2 \end{matrix}\right]$. What operators can you use to put the above equation into it's reduced row echelon form? 

# Put your answer to the above question here.

# ---
# <a name=Under-Defined-Systems></a>
# ## B. Under Defined Systems
# 
# 
# 
# 
# An under-defined system is one that is non-invertible and the number of unknowns is more than the number of knowns. These system often have infinite numbers of possible solutions and solving them involves finding a set of simplified equations that represent all solutions. 
# 
# Often the simplest way to solve an under-defined systems of equations is to extract the solution directly from the reduced row echelon form.  

# &#9989; **<font color=red>QUESTION:</font>**  What is the reduced row echelon form of the matrix $A= \left[ 
# \begin{matrix} 1 & 3 \\ 2 & 6 \end{matrix}\right]$.

# In[ ]:





# &#9989; **<font color=red>QUESTION:</font>**  What are the solutions to the above systems of equations if $b= \left[ 
# \begin{matrix} 10\\ 3 \end{matrix}\right]$?

# In[ ]:





# &#9989; **<font color=red>QUESTION: (assignment specific)</font>**   Write the set of all possible solutions to the below system of linear equations as the sum of two vectors.
# 
# $$ x + 3y -z = -3 $$
# $$ 2x -3y + 4z = 12$$
# $$ 4x + 3y + 2z = 6$$
# 
# 

# In[2]:


##your work


# Latex your vectors here.

# 
# ---
# <a name=Invertible-Systems></a>
# ## C. Invertible Systems
# 
# An invertible system has a square $A$ that is invertible such that all the following properties are true:
# 
# 1. $ A^{-1}A = AA^{-1} = I $
# 1. $(A^{-1})^{-1} = A$
# 2. $(cA)^{-1} = \frac{1}{c}A^{-1}$
# 3. $(AB)^{-1} = B^{-1}A^{-1}$
# 4. $(A^n)^{-1} = (A^{-1})^n$
# 1. $(A^\top)^{-1} = (A^{-1})^\top$  here $A^\top$ is the transpose of the matrix $A$.

# Consider the following system of equations:
# 
# $$\begin{bmatrix}5&-2&2 \\ 4 & -3 &4 \\ 4& -6 &7 \end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}1\\2\\3\end{bmatrix}$$
# 
# 
# 

# In[3]:


import numpy as np
import sympy as sym


# In[4]:


A = np.matrix([[5, -2, 2], [4, -3, 4], [4,-6,7]])
b = np.matrix([[1],[2],[3]])
display(sym.Matrix(A))
display(sym.Matrix(b))


# &#9989; **<font color=red>DO THIS:</font>** Solve the system of equations $Ax=b$.

# In[5]:


#work


# 
# ---
# <a name=Overdefined-systems></a>
# ## D. Overdefined systems
# 
# We also learned solutions to overdefined systems (more equations than unknowns) often do not exist. However, we can estimate a solution using Least Squares fit.  
# 
# Consider the following system of equations:
# 
# $$\begin{bmatrix}5&-2&2 \\ 4 & -3 &4 \\ 4& -6 &7 \\ 6 & 3 & -3\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}=\begin{bmatrix}1\\2\\3\\-1\end{bmatrix}$$
# 
# &#9989; **<font color=red>DO THIS:</font>** Solve the above using LSF. 
# 
# 

# In[6]:


#Put your answer to the above question here.


# In[7]:




---
<a name=System-Properties></a>
## 5. System Properties



The above methods for solving systems of linear equations is only part of the story. We also explored ways to understand properties of linear systems.  Properties such as rank, determinate, eigenvectors and eigenvalues all provide insight into the matrices that are at the core of the systems.  

One problem is that as systems get really large the computational cost of finding a solution can also become large and intractable (i.e. difficult to solve).  We use our understanding of matrix properties and "decompositions" to transform systems into simpler forms so that solving the problem also becomes simple. 

In class tomorrow we will review all of these concepts by looking at methods to solve linear systmes of the form $Ax=b$ using $QR$ decomposition.  When we solve for $Ax=b$ with QR decomposition. We have the following steps:
+ Find the $QR$ decomposition of $A$ such that:
    + $R$ is square upper-triangular matrix
    + The Columns of $Q$ are orthonormal
+ From $QRx=b$, we obtain $Rx =Q^\top b$ 
+ Solve for $x$ using back substitution.


# &#9989; **<font color=red>DO THIS:</font>** Search for a video describing the $QR$ decomposition of a matrix. Try to pick a video that you think does a good job in a short amount of time.  

# In[ ]:


Put a link to the video you found here.


# In[ ]:





# ---
# <a name=Assignment-wrap-up></a>
# ## 6. Assignment wrap up
# 
# 
# Please fill out the form that appears when you run the code below.  **You must completely fill this out in order to receive credit for the assignment!** If you cannont load the form below please try logging in to [spartan365.msu.edu](http://spartan365.msu.edu/) and try running it again, or simply use the direct link provided below. 
# 
# [Direct Link to Microsoft Form](https://forms.office.com/r/n0PEF9xt59)

# &#9989; **<font color=red>QUESTION: (assignment specific)</font>**   Write the set of all possible solutions to the below system of linear equations as the sum of two vectors.
# 
# $$ x + 3y -z = -3 $$
# $$ 2x -3y + 4z = 12$$
# $$ 4x + 3y + 2z = 6$$

# In[ ]:


from IPython.display import HTML
HTML(
"""
<iframe width="640px" height= "480px" src= "https://forms.office.com/Pages/ResponsePage.aspx?id=MHEXIi9k2UGSEXQjetVofSS1ePbivlBPgYEBiz_zsf1UOTk3QU5VVEo1SVpKWlpaWlU4WTlDUlQwWi4u&embed=true" frameborder= "0" marginwidth= "0" marginheight= "0" style= "border: none; max-width:100%; max-height:100vh" allowfullscreen webkitallowfullscreen mozallowfullscreen msallowfullscreen> </iframe>
"""
)


# 
# 
