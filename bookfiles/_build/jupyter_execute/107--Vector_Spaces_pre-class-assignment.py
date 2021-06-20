#!/usr/bin/env python
# coding: utf-8

# 

# # 107 Pre-Class Assignment: Vector Spaces

# ### Readings for this topic (Recommended in bold)
#  * [**_Heffron Chapter  2 II pg 77-86_**](http://joshua.smcvt.edu/linearalgebra/book.pdf)
#  * [Beezer Chapter VS pg 257-269](http://linear.ups.edu/download/fcla-3.50-tablet.pdf)
# 

# ### Goals for today's pre-class assignment 
# 
# 1. [Vector Spaces](#Vector_Spaces)
# 1. [Basis Vectors](#Basis_Vectors)
# 1. [Lots of Things Can Be Vector Spaces](#Examples)
# 1. [Subspaces](#subspaces)
# 1. [Orthogonal_Complement](#Orthogonal_Complement)
# 1. [Assignment Wrap-up](#Assignment_Wrap-up)

# ----
# 
# <a name="Vector_Spaces"></a>
# ## 1.  Vector Spaces
# 
# Vector spaces are an abstract concept used in math. So far we have talked about vectors of real numbers ($R^n$). However, there are other types of vectors as well.  A vector space is a formal definition. If you can define a concept as a vector space then you can use the tools of linear algebra to work with those concepts.  
# 
# A **Vector Space** is a set $V$ of elements called **vectors**, having operations of addition and scalar multiplication defined on it that satisfy the following conditions ($u$, $v$, and $w$ are arbitrary elements of $V$, and $c$ and $d$ are scalars.)
# 
# ### Closure Axioms
# 
# 1. The sum $u + v$ exists and is an element of $V$. ($V$ is closed under addition.)
# 2. $cu$ is an element of $V$. ($V$ is closed under multiplication.)
# 
# ### Addition Axioms
# 
# 3. $u + v = v + u$ (commutative property)
# 4. $u + (v + w) = (u + v) + w$ (associative property)
# 5. There exists an element of $V$, called a **zero vector**, denoted $0$, such that $u+0 = u$
# 6. For every element $u$ of $V$, there exists an element called a **negative** of $u$, denoted $-u$, such that $u + (-u) = 0$.
# 
# ### Scalar Multiplication Axioms
# 
# 7. $c(u+v) = cu + cv$
# 8. $(c + d)u = cu + du$
# 9.  $c(du) = (cd)u$
# 10. $1u = u$
# 
# 

# ### Examples
# 
# Below are a list of the standard examples of vector spaces. Take a second to think about how they satisfy the list of axioms above. 
# 
# - The set of all length 3 vectors form a vector space. Notice that all of the addition and scalar axioms are exactly what we defined previously in the first notebook on vectors. 
# - The symbol $\mathbb R$ means the set of all real numbers, and $\mathbb R ^n$ is the set of all length $n$ vectors whose entries are real numbers. $\mathbb{R}^n$ is a vector space for all $n$.
# - The symbol $\textrm{Mat}_{m \times n}$ represents the set of all $m \times n$ matrices. These also form a vector space. 
# - Vector spaces don't have to necesarily involve matrices and vectors. The set of all 2nd degree polynomials also forms a vector space. Notice that if you have two polynomials $u = 5x^2+3x+1$ and $v = x^2 - x +2$ then $u+v$ is again a 2nd degree polynomial so we satisfy the closue axiom. The others are easily verified aswell. 
# 
# So now that we have some examples of vector spaces. How can we work with the elements in the vector space (=vectors)?

# ----
# <a name="Basis_Vectors"></a>
# ## 1. Basis Vectors
# 
# In order to begin answering the question posed above we need to review/define some concepts.
# 
# Below is a really good video that discusses the concepts of Linear combinations, span, and basis vectors. 

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo("k7RM-ot2NWY",width=640,height=360, cc_load_policy=True)


# &#9989; **<font color=red>QUESTION:</font>** Write three vectors that span $R^3$.
# 

# Put your answer to the above question here

# From the above video two terms we want you to really understand are _**Span**_ and **_Linear Independence_**. Understanding these two concepts are fundamentally important in describing a basis.  Make sure you watch the video and try to answer the following questions as best you can using your own words.  

# &#9989; **<font color=red>QUESTION:</font>** Describe what the _**Span**_ of two vectors is.
# 

# Put your answer to the above question here

# &#9989; **<font color=red>QUESTION:</font>** What is the span of two vectors that point in the same direction?
# 

# Put your answer to the above question here

# &#9989; **<font color=red>QUESTION:</font>** Can the following vectors span $R^3$? Why?
# $$(1,-2,3),\quad (-2,4,-6),\quad (0,6,4)$$

# Put your answer to the above question here

# &#9989; **<font color=red>QUESTION:</font>** What is the **technical definition** of linear independence of vectors?

# Put your answer to the above question here

# **Definition.** A **_Basis_** for a vector space $V$ is any set of vectors in $V$ that are linearly independent and span the vector space $V$.

# ----
# <a name="Examples"></a>
# ## 3. Lots of Things Can Be Vector Spaces

# In[2]:


from IPython.display import YouTubeVideo
YouTubeVideo("YmGWj9RrNMI",width=640,height=360, cc_load_policy=True)


# Consider the following two matrices $A\in \textrm{Mat}_{3x3}$ and $B\in \textrm{Mat}_{3x3}$, which consist of real numbers:

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import sympy as sym
sym.init_printing()

a11,a12,a13,a21,a22,a23,a31,a32,a33 = sym.symbols('a_{11},a_{12}, a_{13},a_{21},a_{22},a_{23},a_{31},a_{32},a_{33}', negative=False)
A = sym.Matrix([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
A


# In[4]:


b11,b12,b13,b21,b22,b23,b31,b32,b33 = sym.symbols('b_{11},b_{12}, b_{13},b_{21},b_{22},b_{23},b_{31},b_{32},b_{33}', negative=False)
B = sym.Matrix([[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]])
B


# &#9989; **<font color=red>QUESTION:</font>** What properties do we need to show all $3\times 3$ matrices of real numbers form a vector space. 

# Put your answer here

# &#9989; **<font color=red>DO THIS:</font>** Demonstrate a few of these properties using **sympy**.

# In[5]:


#Put your answer here. 


# &#9989; **<font color=red>QUESTION (assignment specific):</font>** Determine whether $A$ is a linear combination of $B$, $C$, and $D$?
# 
# $$ A=
# \left[
# \begin{matrix}
#     7 & 6 \\
#     -5 & -3 
# \end{matrix}
# \right],
# B=
# \left[
# \begin{matrix}
#     3 & 0 \\
#     1 & 1 
# \end{matrix}
# \right],
# C=
# \left[
# \begin{matrix}
#     0 & 1 \\
#     3 & 4 
# \end{matrix}
# \right],
# D=
# \left[
# \begin{matrix}
#     1 & 2 \\
#     0 & 1 
# \end{matrix}
# \right]
# $$
# 
# **Hint.** This is equivalent to solving a certain system of equations. Use the definition of linear combination to get started.

# In[6]:


#Put your answer to the above question here


# &#9989; **<font color=red>QUESTION:</font>**  Write a basis for all $2\times 3$ matrices.

# Put your answer to the above question here.

# &#9989; **<font color=red>QUESTION:</font>**  How many elements are in your basis above?

# Put your answer to the above question here.

# **Definition/Theorem.** Any basis for a vector space $V$ will have the same number of elements. The number of elements in a basis for a vector space $V$ is called the dimension of $V$. 

# ----
# <a name="subspaces"> </a>
# ## 4. Subspaces
# 
# Fundamentally vector spaces are just sets that we embue with the additional structure of addition and scalar multiplication. In light of the fact that they are both sets we have the following definition. 
# 
# **Definition** Let $V$ be a vector space. If $W$ is a subset of $V$ and a vector space we say that $W$ is a subspace of $V$. 
# 
# Think about $V = \mathbb R ^3$. There are (infinitely) many different planes (2-dimensional vector spaces) that sit inside of $V$. For example the $xy$-plane, $xz$-plane, and $yz$-plane are three examples of 2-dimensional subspaces of $\mathbb{R}^3$.
# 
# The set of vectors $$W = \textrm{span}\left(\begin{bmatrix} 1 \\ 0 \\1  \end{bmatrix}, \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix} \right)$$ is a fourth example of a 2-dimensional subspace of $\mathbb{R}^3$. (Notice there are two basis vectors of length 3 in the definition of $W$.)
# 
# **Theorem** If $W$ is a subset of a vector space $V$ then $W$ is a subspace of $V$ if
# 
# 1. The vectors in $W$ are closed under addition. This means for any $w_1$ and $w_2$ in $W$ we have that $w_1+w_2$ is in $W$.
# 1. The vectors in $W$ are closed under scalar multiplication. This means for any $w$ in $W$ and $c$ in $\mathbb R$ we have that $cw$ is in $W$.
# 1. $W$ contains the zero vector.

# ----
# <a name="Orthogonal_Complement"></a>
# ## 5. Orthogonal Complement
# 
# **Definition**: A vector $u$ is **orthogonal to a subspace** $W$ of $R^n$ if $u$ is orthogonal to any $w$ in $W$ ($u\cdot w=0$ for all $w\in W$).
# 
# For example, consider the following figure, if we consider the plane to be a subspace then the perpendicular vector comming out of the plane is is orthoginal to any vector in the plane:
# 
# <img alt="Image of a 2D plane with a vector pointing directly out of the surface." src="https://lh5.googleusercontent.com/KC1bkJgC9ihevnOCqeMn_efEdkvgcx5TeBkEVYniwo7T_KxmBu76irZKluAj5PNor9SWdCg4RMS6BZDpNSJOmmz6l6cY0mEc5pq6iR9Qu8AzvWb12lgOO-YUBqiu=w416">
# 
# **Definition**: The **orthogonal complement** of $W$ is the set of all vectors that are orthogonal to $W$. The set is denoted as $W_{\bot}$. 
# 
# For example, if we take $W$ to be the $xy$-plane in $\mathbb R ^3$ then the $z$-axis is its orthogonal complement.

# &#9989; **<font color=red>QUESTION:</font>** Is $W_\bot$ a subspace of $R^n$? Justify your answer briefly.

# Put your answer to the above question here

# &#9989; **<font color=red>QUESTION:</font>** What are the vectors in both $W$ and $W_\bot$?

# Put your answer to the above question here

# ---
# <a name=Assignment-wrap-up></a>
# ## 6. Assignment wrap up
# 
# 
# Please fill out the form that appears when you run the code below.  **You must completely fill this out in order to receive credit for the assignment!** If you cannont load the form below please try logging in to [spartan365.msu.edu](http://spartan365.msu.edu/) and try running it again, or simply use the direct link provided below. 
# 
# [Direct Link to Microsoft Form](https://forms.office.com/r/n0PEF9xt59)
# 
# 
# 

# In[ ]:





# In[7]:


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

# ---
# Written by Dr. Dirk Colbry, and Dr. Matthew Mills Michigan State University
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
# 
# ----

# 
# 
