#!/usr/bin/env python
# coding: utf-8

# 

# 

# # 108 In-Class Assignment: Change of Basis, Projections, Inner Products
# 
# <img alt="Graph showing how one vector can be projected onto another vector by forming a right triangle" src="https://upload.wikimedia.org/wikipedia/commons/9/98/Projection_and_rejection.png" width="50%">
# 
# Image from: https://en.wikipedia.org/wiki/Vector_projection
# 
# 

# ### Agenda for today's class
# 
# 1. [Change of Basis](#CoB)
# 1. [Understanding Projections with Code](#Understanding_Projections_with_Code)
# 1. [Gram-Schmidt Orthoganalization Process](#Gram-Schmidt_Orthoganalization_Process)
# 1. [Subspace Projections](#subspace)
# 1. [The Orthogonal Decomposition Theorem](#orthog_decomp)
# 1. [Inner Products and Matrices](#innerP)
# 1. [Function Approximation](#Function_Approximation)
# 
# 
# 

# ---
# <a name="CoB"></a>
# # 1.Change of Basis
# 
# Here is a quick test to see if you understand how to change coordinate systems for vector spaces. We won't spend a lot of time this semester on this concept, but it is a good question for a quiz/exam. 
# 
# 

# &#9989;  **<font color=red>QUESTION:</font>** Given the basis $S$ for $\mathbb{R}^3$,
# $$S = \{(2,-1,-1), (0,1,3), (1,1,1)\},$$
# 
# Find the coordinate vector of $u=(1,-4,-5)$ relative to the given basis $S$. Store this coordinate in a variable ```ub``` for checking.
# 

# In[1]:


##work here
import numpy as np


# In[2]:


from answercheck import checkanswer
checkanswer.vector(ub,"91cc7b39126e04b42501804cc9ef7d83");


# &#9989;  **<font color=red>QUESTION:</font>** Given two bases $S$ and $S'$ in $\mathbb{R}^3$,
# $$S = \{(2,-1,-1), (0,1,3), (1,1,1)\},$$
# $$S' = \{(1,4,7), (2,5,8), (3,6,10)\},$$
# Find the transition matrix $T$ that will take points in the $S$ coordinate representation and put them into $S'$ coordinates.

# In[ ]:


##work here
import numpy as np


# In[ ]:


from answercheck import checkanswer
checkanswer.matrix(T,"d467bed81305d1623528055cd63e8194");


# ----
# <a name="Understanding_Projections_with_Code"></a>
# # 2. Understanding Projections With Code
# 
# In this in-class assignment, we are going to avoid some of the more advanced libraries ((i.e. no ```numpy``` or ```scipy``` or ```sympy```) to try to get a better understanding about what is going on in the math. 
# The following code implements some common linear algebra functions:

# In[ ]:


#Standard Python Libraries only
import math
import copy


# In[ ]:


def dot(u,v):
    '''Calculate the dot product between vectors u and v'''
    temp = 0;
    for i in range(len(u)):
        temp += u[i]*v[i]
    return temp


# In[ ]:


def multiply(m1,m2):
    '''Calculate the matrix multiplication between m1 and m2 represented as list-of-list.'''
    n = len(m1)
    d = len(m2)
    m = len(m2[0])
    
    if len(m1[0]) != d:
        print("ERROR - inner dimentions not equal")
    result = [[0 for i in range(n)] for j in range(m)]
    for i in range(0,n):
        for j in range(0,m):
            for k in range(0,d):
                result[i][j] = result[i][j] + m1[i][k] * m2[k][j]
    return result


# In[ ]:


def add_vectors(v1,v2):
    v3 = []
    for i in range(len(v1)):
        v3.append(v1[i]+v2[i])
    return v3


# In[ ]:


def sub_vectors(v1,v2):
    v3 = []
    for i in range(len(v1)):
        v3.append(v1[i]-v2[i])
    return v3


# In[ ]:


def norm(u):
    '''Calculate the norm of vector u'''
    nm = 0
    for i in range(len(u)):
        nm += u[i]*u[i]
    return math.sqrt(nm)


# In[ ]:


def transpose(A):
    '''Calculate the transpose of matrix A represented as list of lists'''
    n = len(A)
    m = len(A[0])
    AT = list()
    for j in range(0,m):    
        temp = list()
        for i in range(0,n):
            temp.append(A[i][j])
        AT.append(temp)
    return AT


# ## Projection function
# 
# &#9989; **<font color=red>DO THIS:</font>** Write a function that projects vector $v$ onto vector $u$. 
# Do not use the numpy library. 
# Instead use the functions provided above:
# 
# $$\mbox{proj}_u v = \frac{v \cdot u}{u \cdot u} u$$
# 
# Make sure this function will work for any size of $v$ and $u$. 

# In[ ]:


def proj(v,u):
    ## Put your code here
    return pv


# Let's test your function. Below are two example vectors. Make sure you get the correct answers. 
# You may want to test this code with more than one set of vectors. 

# In[ ]:


u = [1,2,0,3]
v = [4,0,5,8]
print(proj(u,v))


# In[ ]:


from answercheck import checkanswer

checkanswer.vector(proj(u,v),'53216508af49c616fa0f4e9676ce3b9d');


# ### Visualizing projections
# 
# &#9989; **<font color=red>DO THIS:</font>**See if you can design and implement a small function that takes two vectors ($a$ and $b$) as inputs and generates a figure similar to the one above.
# 
# 
# I.e. a black line from the origin to "$b$", a black line from origin to "$a$"; a green line showing the "$a$" component in the "$b$" direction and a red line showing the "$a$" component orthogonal to the green line. 
# Also see section titled "Projection of One Vector onto Another Vector" in Section 4.6 on page 258 of the book.
# 
# When complete, show your solution to the instructor.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt

b = [3,2]
a = [2,3]

def show_projection(a,b):
    plt.plot([0,a[0]], [0,a[1]], color='black')
    plt.annotate('b', b, 
            xytext=(0.9, 0.7), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')
    plt.annotate('a', a, 
            xytext=(0.7, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')
    plt.plot([0,b[0]], [0,b[1]], color='black')
    
#Finish your code here

    plt.axis('equal')
    
x = show_projection(a,b) ;


# ----
# 
# <a name="Gram-Schmidt_Orthoganalization_Process"></a>
# 
# # 3. Gram-Schmidt Orthoganalization Process
# 
# &#9989; **<font color=red>DO THIS:</font>** Implement the Gram-Schmidt orthoganalization process from the [Hefron](http://joshua.smcvt.edu/linearalgebra/book.pdf) textbook (page 282). 
# This function takes a $m \times n$ Matrix $A$ with linearly independent columns as input and return a $m \times n$ Matrix $G$ with orthogonal column vectors. 
# The basic algorithm works as follows:
# 
# - ```AT = transpose(A)``` (this process works with the columns of the matrix so it is easier to work with the transpose. Think about a list of list, it is easy to get a row (a list)).  
# - Make a new empty list of the same size as ```AT``` and call it ```GT``` (G transpose)
# - Loop index ```i``` over all of the rows in AT (i.e. columns of A) 
# 
#     - ```GT[i] = AT[i]```
#     - Loop index ```j``` from 0 to ```i```
#         - ```GT[i] -= proj(GT[i], GT[j])```
#         
#         
# - ```G = transpose(GT)```
# 
# Use the following function definition as a template:

# In[ ]:


def GramSchmidt(A):
    return G


# Here, we are going to test your function using the vectors:

# In[ ]:


A4 = [[1,4,8],[2,0,1],[0,5,5],[3,8,6]]
print(transpose(A4))
G4 = GramSchmidt(A4)
print(transpose(G4))


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(G4,'a472a81eef411c0df03ae9a072dfa040');


# In[ ]:


A2 = [[-4,-6],[3,5]]
print(transpose(A2))
G2 = GramSchmidt(A2)
print(transpose(G2))


# In[ ]:


from answercheck import checkanswer

checkanswer.matrix(G2,'23b9860b72dbe5b84d7c598c08af9688');


# ------
# <a name ='subspace'></a>
# ## 4. Subspace Projections
# 
# The following is the matimatical defination of projection onto a subspace.
# 
# **Definition**: Let $W$ be a subspace of $R^n$ of dimension $m$. Let $\{w_1,\cdots,w_m\}$ be an orthogonal basis for $W$. Then the projection of vector $v$ in $R^n$ onto $W$ is denoted as $\mbox{proj}_Wv$ and is defined as the sum of the projections onto the basis vectors for $W$:
# 
# $$\mbox{proj}_Wv = \mbox{proj}_{w_1} v + \dots + \mbox{proj}_{w_n} v\\
# = \frac{(v\cdot w_1)}{(w_1\cdot w_1)}w_1+\frac{(v\cdot w_2)}{(w_2\cdot w_2)}w_2+\cdots+\frac{(v\cdot w_m)}{(w_m\cdot w_m)}w_m$$
# 
# 
# Another way to say the above definition is that the project of $v$ onto the $W$ is just the sumation of $v$ projected onto each vector in a basis of $W$
# 
# 
# **Remarks**: 
# > Recall in the lecture on *Projections*, we discussed the projection onto a vector, which is the case for $m=1$. We used the projection for $m>1$ in the Gram-Schmidt algorithm. 
# 
# > The projection does not depend on which orthogonal basis you choose. 
# 
# > If $v$ is in $W$, we have $\mbox{proj}_Wv=v$.

# In[ ]:


import numpy as np
import sympy as sym
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
sym.init_printing()


# &#9989; **<font color=red>QUESTION:</font>** Let $v=(3, 2, 6)$ and $W$ is the subspace consisting all vectors with the form $(a, b, b)$. Find the projection of $v$ onto $W$. That is find $w = \mbox{proj}_W v$.

# Put your answer to the above question here

# In[ ]:


##work here


# In[ ]:


from answercheck import checkanswer
checkanswer.vector(w,'13404b60fa3f349ae3982b2587048040')


# <a name = 'orthog_decomp'></a>
# ## 5. The Orthogonal Decomposition Theorem
# **Theorem**: Let $W$ be a subspace of $R^n$. Every vector $v$ in $R^n$ can be written uniquely in the form 
# $$v= w+w_{\bot},$$
# where $w$ is in $W$ and $w_\bot$ is orthogonal to $W$ (i.e., $w_\bot$ is in $W_\bot$). 
# In addition, $w=\mbox{proj}_Wv$, and $w_\bot = v-\mbox{proj}_Wv$.
# 
# **Definition**: Let $x$ be a point in $R^n$, $W$ be a subspace of $R^n$. The distance from $x$ to $W$ is defined to be the minimum of the distances from $x$ to any point $y$ in $W$.
# $$d(x,W)=\min \{\|x-y\|: \mbox{ for all }y \mbox{ in } W\}.$$
# The optimal $y$ can be achieved at $\mbox{proj}_Wx$, and $d(x,W)=\|x-\mbox{proj}_Wx\| = \| x_\bot \|$.

# &#9989; **<font color=red>QUESTION:</font>** Let $v=(3, 2, 6)$ and $W$ is the subspace consisting all vectors with the form $(a, b, b)$. Find the distance from $v$ to $W$.
# 
# **Hint.** Your answer from the previous question will be very helpful to this problem. 

# Put your answer to the above question here

# In[ ]:


##work here

# dist = 


# In[ ]:


from answercheck import checkanswer
checkanswer.float(dist,'f8f24a9cedb159fc084bc4e6e347dc07')


# ---- 
# <a name='innerP'></a>
# ## 6. Inner Products and Matrices
# 
# The following is a review from the pre-class assignment.
# 
# An inner product on a real vector space $V$ is a function that associates a number, denoted as $\langle u,v \rangle$, with each pair of vectors $u$ and $v$ of $V$. This function satisfies the following conditions for vectors $u, v, w$ and scalar $c$:
# 
# 
# $$\langle u,v \rangle = \langle v,u \rangle \text{ Symmetry axiom}$$ 
# 
# $$\langle u+v,w \rangle = \langle u,w \rangle + \langle v,w \rangle \text{ Additive axiom}$$ 
# 
# $$\langle cu,v \rangle = c\langle v,u \rangle \text{ Homogeneity axiom}$$ 
# 
# $$\langle u,u \rangle \ge 0 \text{ and } \langle u,u \rangle = 0 \text{ if and only if } u = 0 \text{ Positive definite axiom}$$ 
# 
# 
# The dot product of $R^n$ is an inner product. However, we can define many other inner products.
# 
# Notice that the dot product of two column vectors $u= \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}$ and $v= \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}$ can also be written as a matrix multiplication 
# 
# $$u \cdot v = u_1v_1 + u_2v_2 = u^\top * v$$
# 
# Now imagine that there is an identity matrix in the middle of the matrix product and we have 
# $u \cdot v = u^\top Iv$. This exmplifies a fundamental relationship between inner products and matrices.
# 
# Now notice for an arbitary matrix $A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$ we have 
# 
# $$ u^\top A v = a_{11} u_1v_1 + a_{12}u_1v_2 + a_{21}u_2v_1 + a_{22}u_2v_2$$ which looks an awful lot like a formula for an inner product. Definining the function $f(u,v) = u^\top A v$ guarentees that it satisfies the additive and homogeneity axiom for inner products. We need a matrix way to represent the other two axioms for inner products.
# 
# **Definition.** A square matrix is said to be positive definite if all of its eigenvalues are positive.
# 
# **Definition** A matrix $A$ is said to be symmetric if $a_{ij} = a_{ji}$ for all entries of the matrix.
# 
# **Definition.** An $n \times n$ matrix $A$ defines an inner product on $R^n$ by the formula $$ \langle u,v \rangle = u^\top A v$$ if and only if the matrix $A$ is positive definite and symmetric.
# 
# 

# &#9989; **<font color=red>Do This:</font>** Show that the matrix $$ A = \begin{bmatrix} 3 & -1 \\ -1 & 2 \end{bmatrix}$$ is positive definite. 

# In[ ]:


##your work here
A = np.matrix('3,-1;-1,2')


# &#9989; **<font color=red>QUESTION:</font>** Does the matrix $A$ given above represent an inner product on $R^2$?

# Your answer here.

# &#9989; **<font color=red>QUESTION:</font>** Find the matrix representing the inner product
# 
# $$\langle u,v \rangle = 6u_1v_1 -2 u_1v_2 -2 u_2 v_1 +4 u_2 v_2 $$ 
# 
# and use it to compute the norm of the vector $ v = [1,-2].$
# 
# **Hint.** Double check the pre-class assignment for the formula for norm.

# In[ ]:


#work here


# In[ ]:


from answercheck import checkanswer
checkanswer.float(norm,'afcf64b47c2dc5ae8e86701f0cbd5eb9')


# ---
# <a name="Function_Approximation"></a>
# ## 7. Function Approximation
# 
# **Definition:** Let $C[a,b]$ be a vector space of all possible continuous functions over the interval $[a,b]$ with inner product:
# $$\langle f,g \rangle = \int_a^b f(x)g(x) dx.$$
# 
# 
# 
# Now let $f$ be an element of $C[a,b]$, and $W$ be a subspace of $C[a,b]$. The function $g \in W$ such that the distance 
# 
# $$d(f,g) = \sqrt{\langle f-g, f-g \rangle} = \int_a^b \left[ f(x) - g(x) \right]^2 dx$$ 
# 
# is a minimum is called the **least-squares approximation** to $f$.
# 
# 
# As stated in the "The Orthogonal Decomposition Theorem" section of this notebook The least-squares approximation to $f$ in the subspace $W$ can be calculated as the projection of $f$ onto $W$, $g = proj_Wf$. Like in the "Subspace Projections" section if we rewrite the formula using an arbitary inner product in lieu of the dot product we have the following:
# 
# If $\{g_1, \ldots, g_n\}$ is an orthogonal basis for $W$, we have 
# 
#  $$prog_Wf = \frac{\langle f,g_1 \rangle}{\langle g_1,g_1 \rangle} g_1 + \dots + \frac{\langle f,g_n \rangle}{\langle g_n,g_n
#  \rangle} g_n$$
#  
#  
# ###  Polynomial Approximations
# 
# An orthogonal bases for all polynomials of degree less than or equal to $n$ can be computed using Gram-schmidt orthogonalization process.  First we start with the following standard basis vectors in $W$
# 
# $$ \{ 1, x, \ldots, x^n \}$$
# 
# The Gram-Schmidt process can be used to make these vectors orthogonal. The resulting polynomials on $[-1,1]$ are called  **Legendre polynomials**.  The first six Legendre polynomial basis elements are:
# 
# $$1$$
# $$x$$
# $$x^2 -\frac{1}{3}$$
# $$x^3 - \frac{3}{5}x$$
# $$x^4 - \frac{6}{7}x^2 + \frac{3}{35}$$
# $$x^5 - \frac{10}{9}x^3 + \frac{5}{12}x$$

# &#9989;**<font color=red>QUESTION:**</font> What is the least-squares linear approximations of $f(x) = e^x$ over the interval $[-1, 1]$. In other words, what is the projection of $f$ onto $W$, where $W$ is a first order polynomal with basis vectors $\{1, x\} (i.e. n=1)$. 
# 
# **Hint.** You can give the answer in integrals without computing the integrals. Note the Legendre polynomials are not normalized.

# Put your answer to the above question here.

# Here is a plot of the equation $f(x) = e^x$:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np

#px = np.linspace(-1,1,100)
#py = np.exp(px)
#plt.plot(px,py, color='red');
import sympy as sym
from sympy.plotting import plot
x = sym.symbols('x')
f = sym.exp(x)
plot(f,(x,-1,1))


# We can use `sympy` to compute the integral. The following code compute the definite integral of 
# $$\int_{-1}^1 e^x dx.$$
# In fact, `sympy` can also compute the indefinite integral by removing the interval.

# In[ ]:


sym.init_printing()
x = sym.symbols('x')
sym.integrate('exp(x)',(x, -1, 1))
#sym.integrate('exp(x)',(x))


# Use `sympy` to compute the first order polynomial that approximates the function $e^x$.
# The following calculates the above approximation written in ```sympy```:

# In[ ]:


g_0 = sym.integrate('exp(x)*1',(x, -1, 1))/sym.integrate('1*1',(x,-1,1))*1
g_1 = g_0 + sym.integrate('exp(x)*x',(x,-1,1))/sym.integrate('x*x',(x,-1,1))*x
g_1 


# Plot the original function $f(x)=e^x$ and its approximation.

# In[ ]:


p2 = plot(f, g_1,(x,-1,1))


# In[ ]:


#For fun, I turned this into a function:
x = sym.symbols('x')

def lsf_poly(f, gb = [1,  x], a =-1, b=1):
    proj = 0
    for g in gb:
#        print(sym.integrate(g*f,(x,a,b)))
        proj = proj + sym.integrate(g*f,(x,a,b))/sym.integrate(g*g,(x,a,b))*g
    return proj

lsf_poly(sym.exp(x))


# &#9989;**<font color=red>QUESTION:</font>** What would a second order approximation look like for this function? How about a fifth order approximation?

# Put your answer to the above question here

# In[ ]:


x = sym.symbols('x')
g_2 = 
g_2


# In[ ]:


p2 = plot(f, g_2,(x,-1,1))


# 
# 
