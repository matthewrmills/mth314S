#!/usr/bin/env python
# coding: utf-8

# 

# 

# # 107 In-Class Assignment: Vector Spaces & Fundamental Spaces of a Matrix
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/3d_basis_transformation.svg/580px-3d_basis_transformation.svg.png" width="50%">
# 
# Image from: [https://en.wikipedia.org/wiki/Change_of_basis](https://en.wikipedia.org/wiki/Change_of_basis)
# 
# 
# 
#     

# ### Agenda for today's class
# 
# 
# 1. [Basis Vectors](#Basis-Vectors)
# 1. [Vector Spaces](#Vector-Spaces)
# 1. [Four Fundamental Spaces](#Four_Fundamental_Subspaces)
# 1. [Practice Example](#Practice_Example)
# 1. [Practice Example](#Practice_Example2)
# 
# 

# 
# 
# ---
# <a name=Basis-Vectors></a>
# ## 1. Basis Vectors
# 
# 

# Consider the following example. We claim that the following set of vectors form a baiss for $R^3$:
# 
# $$B = \{(2,1, 3), (-1,6, 0), (3, 4, -10) \}$$

# If these vectors form a basis they must be _**linearly independent**_ and _**Span**_ the entire space of $R^3$

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import sympy as sym
sym.init_printing(use_unicode=True)


# &#9989; **<font color=red>DO THIS:</font>** Create a $3 \times 3$ numpy matrix $A$ where the columns of $A$ form are the basis vectors. 

# In[2]:


#Put your answer to the above question here


# In[3]:


from answercheck import checkanswer

checkanswer.matrix(A,'68b81f1c1041158b519936cb1a2e4d6b');


# &#9989; **<font color=red>DO THIS:</font>** Using python, calculate the determinant of matrix $A$.

# In[ ]:


# Put your answer to the above question here. 


# &#9989; **<font color=red>DO THIS:</font>** Using python, calculate the inverse of $A$.

# In[ ]:


# Put your answer to the above question here.


# &#9989; **<font color=red>DO THIS:</font>** Using python, calculate the rank of $A$.

# In[ ]:


# Put your answer to the above question here.


# &#9989; **<font color=red>DO THIS:</font>** Using python, calculate the reduced row echelon form of $A$.

# In[ ]:


# Put your answer to the above question here. 


# &#9989; **<font color=red>DO THIS:</font>** Using the above $A$ and the vector $b=(1,3,2)$.  What is the solution to $Ax=b$?  

# In[ ]:


#Put your answer to the above question here.


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(x,'8b0938260dfeaafc9f8e9fec0bc72f17');


# Turns out a matrix where column vectors are formed from basis vectors a lot of interesting properties and the following statements are equivalent.
# 
# - The column vectors of $A$ form a basis for $R^n$
# - $|A| \ne 0$
# - $A$ is invertible.
# - $A$ is row equivalent to $I_n$ (i.e. it's reduced row echelon form is $I_n$)
# - The system of equations $Ax = b$ has a unique solution.
# - $rank(A) = n$
# 

# Not all matrices follow the above statements but the ones that do are used throughout linear algebra so it is important that we know these properties. 

# 
# 
# ---
# <a name=Vector-Spaces></a>
# ## 2. Vector Spaces
# 
# A **Vector Space** is a set $V$ of elements called **vectors**, having operations of addition and scalar multiplication defined on it that satisfy the following conditions ($u$, $v$, and $w$ are arbitrary elements of $V$, and c and d are scalars.)
# 
# ### Closure Axioms
# 
# 1. The sum $u + v$ exists and is an element of $V$. ($V$ is closed under addition.)
# 2. $cu$ is an element of $V$. ($V$ is closed under scalar multiplication.)
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
# ### Definition of a basis of a vector space
# 
# > A finite set of vectors ${v_1,\dots, v_n}$ is called a **basis** of a *vector space* $V$ if the set *spans* $V$ and is *linearly independent*. 
# >i.e. each vector in $V$ can be expressed uniquely as a *linear combination* of the vectors in a basis.
# 
# 

# ## Vector spaces
# 
# &#9989; **<font color=red>DO THIS:</font>** Let $U$ be the set of all circles in $R^2$ having center at the origin. 
# Interpret the origin as being in this set, i.e., it is a circle center at the origin with radius zero. 
# Assume $C_1$ and $C_2$ are elements of $U$. 
# Let $C_1 + C_2$ be the circle centered at the origin, whose radius is the sum of the radii of $C_1$ and $C_2$. 
# Let $kC_1$ be the circle center at the origin, whose radius is $|k|$ times that of $C_1$. 
# Determine which vector space axioms hold and which do not. 
# 

# Put your answer here

# ### Spans:
# 
# &#9989; **<font color=red>DO THIS:</font>** Let $v$, $v_1$, and $v_2$ be vectors in a vector space $V$. 
# Let $v$ be a linear combination of $v_1$ and $v_2$. 
# If $c_1$ and $c_2$ are nonzero real numbers, show that $v$ is also a linear combination of $c_1v_1$ and $c_2v_2$.

# Put your answer here

# &#9989; **<font color=red>DO THIS:</font>** Let $v_1$ and $v_2$ span a vector space $V$. 
# Let $v_3$ be any other vector in $V$. 
# Show that $v_1$, $v_2$, and $v_3$ also span $V$.

# Put your answer here

# ### Linear Independent:
# Consider the following matrix, which is in the reduced row echelon form.
# 
# 
# $$ 
# \left[
# \begin{matrix}
#     1   & 0 & 0 & 7  \\
#     0   & 1 & 0 & 4  \\
#     0   & 0 & 1 & 3
# \end{matrix}
# \right] 
# $$
# 
# &#9989; **<font color=red>DO THIS:</font>** Show that the row vectors form a linearly independent set:
# 

# Put your answer here

# &#9989; **<font color=red>DO THIS:</font>** Is the set of nonzero row vectors of any matrix in reduced row echelon form linearly independent? Discuss in your groups and include your thoughts below.

# Put your answer here

# &#9989; **<font color=red>DO THIS:</font>** A computer program accepts a number of vectors in $R^3$ as input and checks to see if the vectors are linearly independent and outputs a True/False statment. 
# Discuss in your groups, which is more likely to happen due to round-off error--that the computer states that a given set of linearly independent vectors is linearly dependent, or vice versa? 
# Put your groups thoughts below.

# Put your answer here

# ----

# 

# ---
# <a name="Four_Fundamental_Subspaces"></a>
# ## 3. Four Fundamental Subspaces
# 
# <img alt="Classic picture of the four fundamental spaces. Please see text for detailed description." src="https://kevinbinz.files.wordpress.com/2017/02/linear-algebra-fundamental-space-interpretation-6.png" width="100%">
# 
# Image from: https://kevinbinz.com/2017/02/20/linear-algebra/
# 
#     
# 
# ### The four fundamental subspaces
# 
# * Columnspace, $\mathcal{C}(A)$
# * Nullspace, $\mathcal{N}(A)$
# * Rowspaces, $R(A)$
#     * All linear combinations of rows
#     * All the linear combinations of the colums of $A^\top$, $\mathcal{C}(A^\top)$
# * Nullspace of $A^\top$, $\mathcal{N}(A^\top)$ (the left nullspace of $A$)
# 
# ### Where are these spaces for a $m\times n$ matrix $A$?
# * $\mathcal{R}(A)$ is in $R^n$
# * $\mathcal{N}(A)$ is in $R^n$
# * $\mathcal{C}(A)$ is in $R^m$
# * $\mathcal{N}(A^\top)$ is in $R^m$
# 
# ### Calculating basis and dimension
# 
# #### For $\mathcal{R}(A)$
# * If $A$ undergoes row reduction to row echelon form $B$, then $\mathcal{C}(B)\neq \mathcal{C}(A)$, but $\mathcal{R}(B) = \mathcal{R}(A)$ (or $\mathcal{C}(B^\top) = \mathcal{C}(A^\top))$
# * A basis for the rowspace of $A$ (or $B$) is the first $r$ rows of $B$
#     * So we row reduce $A$ and take the pivot rows and transpose them
# * The dimension is also equal to the rank $r$
# 
# #### For $\mathcal{N}(A)$
# * The bases are the special solutions (one for every free variable, $n-r$)
# * The dimension is $n- r$
# 
# 
# #### For $\mathcal{C}(A) = \mathcal{R}(A^\top)$
# * Apply the row reduction on the transpose $A^\top$.
# * The dimension is the rank $r$
# 
# 
# #### For $\mathcal{N}(A^\top)$
# * It is also called the left nullspace, because it ends up on the left (as seen below)
# * Here we have $A^\top y = 0$
#     * $y^\top(A^\top)^\top = 0^\top$
#     * $y^\top A = 0^\top$
#     * This is (again) the special solutions for $A^\top$ (after row reduction)
# * The dimension is $m - r$

# ----
# 
# <a name="Practice_Example"></a>
# ## 4.  Practice Examples:
# 
# Consider the linear transformation defined by the following matrix $A$.  
# 
# $$A = 
# \left[
# \begin{matrix}
#     1 & 2 & 3 & 1  \\
#     1 & 1 & 2 & 1  \\
#     1 & 2 & 3 & 1 
#  \end{matrix}
# \right] 
# $$

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import sympy as sym
sym.init_printing()


# **&#9989;  <font color=red>Question:</font>** What is the reduced row echelon form of $A$?  You can use sympy.

# In[ ]:


#Put your answer to the above question here.


# **&#9989;  <font color=red>Question:</font>** Now let's calculate the row space of $A$. 
# Note that the row space is defined by a linear combination of the non-zero row vectors in the reduced row echelon matrix:

# Put your answer to the above question here

# **Definition** Let $A$ be a matrix. The **homogeneous equation** for $A$ is the system of linear equations $Ax=0$.
# 
# &#9989;  **<font color=red>Question:</font>** Using the reduced row echelon form define the leading variables in terms of the free variables for the homogeneous equation. 

# Put your answer to the above question here

# &#9989;  **<font color=red>Question:</font>** The solution to the above question defines the nullspace of $A$ (aka the Kernel). Use the ``sympy.nullspace`` function to verify your answer.

# In[ ]:


# Put your code here


# &#9989;  **<font color=red>Question:</font>** How do the vectors from your last function call relate to your solution to the homogeneous eqution?

# Your answer here.
# 

# &#9989;  **<font color=red> Question:</font>** Now let's calculate the range of $A$ (column space of $A$).  Note that the range is spanned by the column vectors of $A$. 
# Transpose $A$ and calculate the reduced row echelon form of the transposed matrix like we did above.

# In[ ]:


## Put your code here


# &#9989;  **<font color=red>Question:</font>** The nonzero row vectors of the above solution will give a basis for the range (or $\mathcal{C}(A)$). Write the range of $A$ as a linear combination of these nonzero vectors:

# Put your answer to the above question here.

# &#9989;  **<font color=red>Question:</font>** Finally, using the reduced row echelon form for $A^\top$ define the leading variables in terms of the free variables and define the null space of $A^\top$.

# Put your answer to the above question here.

# &#9989; **<font color=red>DO THIS:</font>**
# Pick one vector from the row space of $A$ and another vector from the null space of $A$. 
# Find the dot product of these two vector.

# In[ ]:


#Put your answer here


# &#9989; **<font color=red>Question:</font>** Did you get the same value for the dot product as other people in your group? Did you choose the same vectors as the otehr people in your group? Explain your answer. 

# Put your answer to the above question here

# ---
# ### Example #2
# 
# Consider the following system of linear equations.
# 
# $$ x_1 - x_2 + x_3 = 3 $$
# $$ -2x_1 + 2x_2 - 2x_3 = -6 $$

# &#9989; **<font color=red>DO THIS:</font>** What are the solutions to the above system of equations? Write your set of solutions as a linear combination of vectors.

# In[ ]:


# Put your code here


# &#9989; **<font color=red>DO THIS:</font>** Choose arbitrary values for your free variables to obtain a specific solution (any solution will do) to the above set of equations.

# Put your answer to the above question here.

# &#9989; **<font color=red>DO THIS:</font>** Now consider only the left hand side of the above matrix and solve for the kernel (null Space) of A:
# 
# 
# $$ A = 
# \left[
# \begin{matrix}
#     1 & -1 & 1  \\
#     -2 & 2 & -2  
# \end{matrix}
# \right] 
# $$

# In[ ]:


#Put your answer here


# &#9989; **<font color=red>Question:</font>** How many free variables do you have in the solution to the system equations? What is the dimension of the nullspace of $A$? 
# 
# **Hint.** These two values should be the same.

# In[ ]:





# **Theorem.** The difference of any two specific solutions to the system of equations $Ax=b$ is an element of the nullspace of $A$.

# ----
# <a name="Practice_Example2"></a>
# ## 5. Practice Nutrition
# 
# Big Annie's Mac and Cheese fans want to improve the levels of protein and fiber for lunch by adding broccoli and canned chicken. 
# The nutrition information for the foods in this problem are 
# 
# 
# |Nutrient    | Mac and Cheese           |  Broccoli        |    Chicken   | Shells and White Cheddar |
# |----|-----------------|----------------|----------|----------|
# |Calories| 270 | 51 |  70 | 260 |
# |Protein (g) | 10 | 5.4 |  15| 9|
# |Fiber (g)| 2   |  5.2 |  0| 5|
# 
# 
# <img alt="Logo for Annie's Mac and Cheese" src="https://upload.wikimedia.org/wikipedia/commons/c/cd/Annies_logo.png" width="50%">
# 
# She wants to achieve the goals with exactly 400 calories, 30 g of protein, and 10 g of fiber by choosing the combination of these three or four serving. (Assume that we can have non-integer proportions for each serving.)
# 
# 

# &#9989; **<font color=red>Question a</font>**: We consider all **four** choices of food together. Formulate the problem into a system of equations 
# $$Ax = b.$$
# Create your matrix $A$ and the column vector $b$ in ``np.matrix``.

# In[ ]:


##your code here


# &#9989; **<font color=red>Question b</font>**: In this and next question, we only consider **three** out of the four choices. What proportions of these servings of the **three** food (Mac and Cheese, Broccoli, and Chicken) should be used to meet the goal? (Hint: formulate it as a system of equations and solve it).

# In[ ]:


#Put your answer here


# &#9989; **<font color=red>Question c</font>**: She found that there was too much broccoli in the proportions from part (b), so she decided to switch from classical Mac and Cheese to Annie’s Whole Wheat Shells and White Cheddar. What proportions of servings of the new **three** food should she use to meet the goals?

# In[ ]:


#Put your answer here


# &#9989; **<font color=red>Question d</font>**: Based on the solutions to parts (b) and (c), what are the possible proportions of serving for the **four** food that meet the goal. 

# Put your answer here

# &#9989; **<font color=red>Question e</font>**: Solve the system of equations from part (a). You need to first decide the three outcomes: One solution, None solution, Infinite many solutions. Then for *One solution*, write down the solution; for *Infinite many solutions*, write down all the solutions with free variables. 

# In[ ]:


#Put your answer here


# Put your answer here

# 
# 
