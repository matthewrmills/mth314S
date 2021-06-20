#!/usr/bin/env python
# coding: utf-8

# 

# 

# 
# # 106 In-Class Assignment: Diagonalization & Linear Dynamical Systems 
# 
# <img alt="Classig equation for diagonalizing a matrix. Will be discussed in class" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/62ab0ef52ecb1e1452efe6acf096923035c75f62" width="50%">
# 
# Image from: [https://en.wikipedia.org/wiki/Diagonalizable_matrix](https://en.wikipedia.org/wiki/Diagonalizable_matrix)
# 
# 
# Today in class we will discuss two applications of eigenvalues and eigenvectors. The first example is called **diagonalization of a matrix** it is a way to decompose a matrix into the product of an invertible matrix, its inverse, and a diagonal matrix. We will see that the eigenvectors and eigenvalues play an important role in constructing these pieces. 
# 
# The second example is where we can use the steadystate eigenvector to solve Markov models, and the diagonalization process to solve systems of linear differential equations. 
# 
# The last section talks about how to turn any graph into a random walk markov model. 
#     

# ### Agenda for today's class
# 
# 1. [Diagonalization](#Diagonalization)
# 1. [The Power of a Matrix](#The_Power_of_a_Matrix)
# 1. [Epidemic Dynamics - Discrete Case](#Epi_discrete)
# 1. [Epidemic Dynamics - Continuous Model](#Epi_continuous)
# 1. [Population Dynamics](#Population)
# 1. [Random Walks from Graphs](#random_walk)
# 

# ----
# <a name="Diagonalization"></a>
# ## 1. Diagonalization

# **_Reminder_**: The eigenvalues of triangular (upper and lower) and diagonal matrices are easy:
# 
# * The eigenvalues for triangular matrices are the diagonal elements.
# * The eigenvalues for the diagonal matrices are the diagonal elements. 

# ### Diagonalization
# 
# 
# **Definition**: A square matrix $A$ is said to be *diagonalizable* if there exist a matrix $C$ such that $D=C^{-1}AC$ is a diagonal matrix.
# 
# **Definition**: $B$ is a *similar matrix* of $A$ if we can find $C$ such that $B=C^{-1}AC$.
# 
# 
# Given an $n\times n$ matrix $A$, can we find another $n \times n$ invertable matrix $C$ such that when $D=C^{-1}AC$ is diagonal, i.e., $A$ is diagonalizable?

# * Because $C$ is inveritble, we have 
# $$C^{-1}AC=D \\ CC^{-1}AC = CD\\ AC = CD $$
# 
# 
# * Generate $C$ as the columns of $n$ linearly independent vectors $(x_1...x_n)$ We can compute $AC=CD$ as follows:
# $$ A\begin{bmatrix} \vdots  & \vdots  & \vdots  & \vdots  \\ \vdots  & \vdots  & \vdots  & \vdots  \\ { x }_{ 1 } & { x }_{ 2 } & \dots  & { x }_{ n } \\ \vdots  & \vdots  & \vdots  & \vdots  \end{bmatrix}=AC=CD=\begin{bmatrix} \vdots  & \vdots  & \vdots  & \vdots  \\ \vdots  & \vdots  & \vdots  & \vdots  \\ { x }_{ 1 } & { x }_{ 2 } & \dots  & { x }_{ n } \\ \vdots  & \vdots  & \vdots  & \vdots  \end{bmatrix}\begin{bmatrix} { \lambda  }_{ 1 } & 0 & 0 & 0 \\ 0 & { \lambda  }_{ 2 } & 0 & 0 \\ \vdots  & \vdots  & { \dots  } & \vdots  \\ 0 & 0 & 0 & { \lambda  }_{ n } \end{bmatrix}$$
# * Then we check the corresponding columns of the both sides. We have 
# $$Ax_1 = \lambda_1x_1\\\vdots\\Ax_n=\lambda x_n$$
# 
# * $A$ has $n$ linear independent eigenvectors.
# 
# * $A$ is said to be *similar* to the diagonal matrix $D$, and the transformation of $A$ into $D$ is called a *similarity transformation*.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import sympy as sym
sym.init_printing()


# ### A simple example
# 
# Consider the following:
# $$ A = \begin{bmatrix}7& -10\\3& -4\end{bmatrix},\quad C = \begin{bmatrix}2& 5\\1& 3\end{bmatrix}$$

# &#9989;  **<font color=red>Do this:</font>** Verify that $C^{-1}AC$ is a diagonal matrix. Define $D = C^{-1}AC.$

# In[2]:


#Put your answer to the above question here.


# In[3]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer.matrix(D, '8313fe0f529090d6a8cdb36248cfdd6c');


# &#9989;  **<font color=red>Do this:</font>** Find the eigenvalues and eigenvectors of $A$. Set variables ```e1``` and ```vec1``` to be the largest eigenvalue and it's associated eigenvector and ```e2, vec2``` to represent the  smallest.

# In[ ]:


#Put your answer to the above question here.


# In[ ]:


from answercheck import checkanswer
checkanswer.float(e1, "d1bd83a33f1a841ab7fda32449746cc4");


# In[ ]:


from answercheck import checkanswer
checkanswer.float(e2, "e4c2e8edac362acab7123654b9e73432");


# In[ ]:


from answercheck import checkanswer
checkanswer.eq_vector(vec1, "09d9df5806bc8ef975074779da1f1023", decimal_accuracy = 4)


# In[ ]:


from answercheck import checkanswer
checkanswer.eq_vector(vec2, "d28f0a721eedb3d5a4c714744883932e", decimal_accuracy = 4)


# &#9989;  **<font color=red>Question:</font>**
# What relationship do you notice between the values of ``e1``, ``e2``, and the matrix $D$ above?

# Your answer here.

# ### A second example
# 
# &#9989;  **<font color=red>Do this:</font>** Consider 
# $$ A = \begin{bmatrix}-4& -6\\3& 5\end{bmatrix}.$$
# Find a matrix $C$ such that $C^{-1}AC$ is diagonal. (Hint, use the function `diagonalize` in `sympy`.)

# In[ ]:


#Put your answer to the above question here. 


# In[ ]:


#Check the output type
assert(type(C)==sym.Matrix)


# In[ ]:


from answercheck import checkanswer
checkanswer.matrix(C,'ba963b7fef354b4a7ddd880ca4bac071')


# ### The third example
# 
# &#9989;  **<font color=red>Do this:</font>** Consider 
# $$ A = \begin{bmatrix}5& -3\\3& -1\end{bmatrix}.$$
# Can we find a matrix $C$ such that $C^{-1}AC$ is diagonal?  (Hint: find eigenvalues and eigenvectors using `sympy`)

# In[ ]:


#Put your answer to the above question here. 


# ### Dimensions of eigenspaces and diagonalization
# 
# We will talk about vector spaces next week. This section will be more relevent then. In the meantime, we will say that a matrix is diagonalizable if the number of eigenvectors it has is equal to the number of columns. 
# 
# **Definition**: The set of all eigenvectors of a $n\times n$ matrix corresponding to a eigenvalue $\lambda$, together with the zero vector, is a subspace of $R^n$. This subspace spaces is called *eigenspace*.
# 
# * For the third example, we have that the characteristic equation $(\lambda-2)^2=0$.
# * Eigenvalue $\lambda=2$ has multiplicity 2, but the eigenspace has dimension 1, since we can not find two lineare independent eigenvector for $\lambda =2$. 
# 
# > The dimension of an eigenspace of a matrix is less than or equal to the multiplicity of the corresponding eigenvalue as a root of the characteristic equation.
# 
# > A matrix is diagonalizable if and only if the dimension of every eigenspace is equal to the multiplicity of the corresponding eigenvalue as a root of the characteristic equation.

# ### The fourth example
# 
# &#9989;  **<font color=red>Do this:</font>** Consider 
# $$ A = \begin{bmatrix}2& -1\\1& 2\end{bmatrix}.$$
# Can we find a matrix $C$ such that $C^{-1}AC$ is diagonal?

# In[ ]:


#Put your answer to the above question here. 


# The take aways that you should get from these examples are the following:
# 
# 1. The $A$ matrix and the $D$ matrix have the same eigenvalues. Specifically the entries of the $D$ matrix are the eigenvalues of $A$.
# 3. The $C$ matrix is the matrix whose columns are the eigenvalues (in the same order as the entries of $D$.)
# 2. Not all matrices are diagonalizable. If there aren't enough linearly independent eigenvalues, the matrix won't be diagonalizable. 
# 4. Sometimes eigenvalues and eigenvectors can take complex values.

# ---
# 
# <a name="The_Power_of_a_Matrix"></a>
# ## 2. The Power of a Matrix
# 
# For a diagonal matrix it is very efficicent to compute a power of the matrix. Using diagonalization we can speed up matrix powers for non-diagonal matrices.
# 
# * For a diagonalizable matrix $A$, we have $C^{-1}AC=D$. Then we have 
# $$A = C D C^{-1}$$
# * We have 
# $$A^2 = C D C^{-1} C D C^{-1} = C D^2 C^{-1}$$
# $$A^n = C D C^{-1} \dots C D C^{-1} = C D^n C^{-1}$$
# * Because the columns of $C$ are eigenvectors, so we can say that the eigenvectors for $A$ and $A^n$ are the same if $A$ is diagonalizable. 
# * If $x$ is an eigenvector of $A$ with the corresponding eigenvalue $\lambda$, then $x$ is also an eigenvector of $A^n$ with the corresponding eigenvalue $\lambda^n$.

# ---
# <a name="Epi_discrete"></a>
# ## 3. Epidemic Dynamics - Discrete Case
# 
# The dynamics of infection and the spread of an epidemic can be modeled as a linear dynamical system. 
# 
# We count the fraction of the population in the following four groups:
# + Susceptible: the individuals can be infected next day
# + Infected: the infected individuals 
# + Recovered (and immune): recovered individuals from the disease and will not be infected again
# + Decreased: the individuals died from the disease
# 
# We denote the fractions of these four groups in $x(t)$. For example $x(t)=(0.8,0.1,0.05,0.05)$ means that at day $t$, 80\% of the population are susceptible, 10% are infected, 5% are recovered and immuned, and 5% died.
# 
# We choose a simple model here. After each day,
# 
# + 5% of the susceptible individuals will get infected 
# + 3% of infected inviduals will die
# + 10% of infected inviduals will recover and immuned to the disease
# + 4% of infected inviduals will recover but not immuned to the disease
# + 83% of the infected inviduals will remain
# 
# 
# 

# In[ ]:


A = np.matrix([[0.95, 0.04, 0, 0],[0.05, 0.83, 0, 0],[0, 0.1, 1, 0],[0,0.03,0,1]])
sym.Matrix(A)


# &#9989;  **<font color=red>Do this:</font>** If we start with $x(0) = (1, 0, 0, 0)$ for day 0. Use the `for` loop to find the distribution of the four groups after 50 days.

# In[ ]:


x0 = np.matrix([[1],[0],[0],[0]])
x  = x0
for i in range(50):
    x = A*x
print(x)


# &#9989;  **<font color=red>Do this:</font>** Finish the program below to apply the above transformation matrix for 200 iterations and plot the results.

# In[ ]:


#Put your answer to the above question here

x0 = np.matrix([[1],[0],[0],[0]])
n = 200 
x  = x0
x_all = np.matrix(np.zeros((4,n)))
for i in range(n):
    ###YOUR CODE BELOW HERE - should be 2 lines.
    
    
    ###YOUR CODE ABOVE HERE
    
for i in range(4):
    plt.plot(x_all[i].T)
    
    


# ---
# <a name="Epi_continuous"></a>
# ## 4. Epidemic Dynamics - Continuous Model
# 
# Instead of using the discrete markov model, we can also use a continuous model with ordinary differential equations. 
# 
# For example, we have that 
# 
# $$\dot{x}_1 = {dx_1(t)\over dt} = -0.05x_1(t)+ 0.04 x_2(t)$$
# It means that the changes in the susceptible group depends on susceptible and infected individuals. It increase because of the recovered people from infected ones and it decreases because of the infection. 
# 
# Similarly, we have the equations for all three groups.
# $$\dot{x}_2 = {dx_2(t)\over dt} = 0.05x_1(t)-0.17 x_2(t) \\ 
# \dot{x}_3 = {dx_3(t)\over dt}= 0.1 x_2(t) \\
# \dot{x}_4 = {dx_4(t)\over dt} = 0.03 x_2(t)$$
# 
# 

# &#9989;  **<font color=red>Do this:</font>** We can write it as system of ODEs as 
# $$\dot{x}(t) = Bx(t)$$
# Write down the matrix $B$ in `numpy.matrix`

# In[ ]:


# Put your answer to the above question here.


# In[ ]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer.matrix(B,'a4a21d2187eef96603e5104df2cc6e61')


# &#9989;  **<font color=red>Question</font>**  What relationship do you see between the $A$ matrix of the discrete case and the $B$ matrix of the continuous case?

# Your answer here

# ---
# The continuous model with differential equations requires diagonalization to solve.
# 
# First note that if we have a regular differential equation like $\dot x = a x$, then a solution will be $x(t) = e^{at}$, in fact for ant constant $c$, the function $x(t) = ce^{at}$ is a solution. 
# 
# In our case we have 
# 
# $$\dot{x}_1 = {dx_1(t)\over dt} = -0.05x_1(t)+ 0.04 x_2(t)$$
# $$\dot{x}_2 = {dx_2(t)\over dt} = 0.05x_1(t)-0.17 x_2(t) \\ 
# \dot{x}_3 = {dx_3(t)\over dt}= 0.1 x_2(t) \\
# \dot{x}_4 = {dx_4(t)\over dt} = 0.03 x_2(t)$$
# 
# Which gives us the matrix equation $$\begin{pmatrix} \dot{x}_1\\ \dot{x}_2 \\ \dot{x}_3\\ \dot{x}_4\end{pmatrix}= \begin{pmatrix} -0.05& 0.04& 0& 0\\0.05& -0.17& 0& 0\\0&0.1& 0& 0\\0&0.03&0&0\end{pmatrix}\begin{pmatrix}{x_1}\\{x_2} \\ {x_3}\\{x_4}\end{pmatrix}$$
# 
# Now if we diagonalize $B$, we have $D = C^{-1}BC$, so if we define $Cy = x$, and so $C\dot y = \dot x$ and we have that 
# 
# $$\dot x = Bx \Rightarrow C\dot y = BCy \Rightarrow  \dot y = C^{-1}BCy = Dy$$
# 
# So we can solve $\dot y = D y$ easier than before since D is diagonal with entries $d_1,d_2,d_3,d_4$ we get the system of equations $\dot{y}_i = d_i y_i$ for $i = 1,2,3,4.$ As we saw these have an easy soltion of using the exponentials. Then once we find the vector $y = (y_1,y_2,y_3,y_4)^T$ we can use the equation $Cy = x$ to solve for the vector $x$.
# 
# Specially we get a solution of,
# $$x_i = c_{i,1} e^{d_1t} + \dots + c_{i,n} e^{d_nt}$$
# 
# 

# &#9989;  **<font color=red>Do This: </font>** **Use ``numpy``** to find matrices $C$ and $D$ such that $D = C^{-1}BC$. Save them as ``C`` and ``D``, respetively.

# In[ ]:


## Your code here


# In[ ]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer(np.allclose(D,np.linalg.inv(C)*B*C),'f827cf462f62848df37c5e1e94a4da74')


# The following code implements a numerical way to solve the continuous model and plots the first 200 days.

# In[ ]:


x0 = np.matrix([[1],[0],[0],[0]])
n = 200
x_all = np.matrix(np.zeros((4,n)))

y0 = np.linalg.inv(C)*x0 ## THE C matrix here should be the from the diagonalization of $B$.
DD = np.diag(np.exp(np.diag(D)))
for i in range(n):
    y = np.matrix(DD)**i*y0
    x = C*y
    x_all[:,i] = x[:,0]    

labels = ["Susceptible","Infected","Immune","Deceased"]
for i in range(4):
    plt.plot(x_all[i].T,label=labels[i])
plt.legend()


# &#9989;  **<font color=red>Do This: </font>** If you have any questions about the code above talk to your instructor.

# ---
# <a name="Population"></a>
# ## 5. Population Dynamics 
# 
# Not all dynamical systems need to be solved using steadystate. Some of them just model different things and the matrix calculations gives a compact way to express the quantities.
# 
# In this section, we consider the distribution of a population at different ages. 
# 
# - Let $x(t)$ be a 100-vector with $x_i(t)$ denoting the number of people with age $i-1$. 
# 
# - The birth rate is given in the vector $b$, where $b_i$ is the average number of births per person with age $i-1$. $b_i=0$ for $i<13$ and $i>50$.  
# 
# - The death rate is given by the vector $d$, where $d_i$ is the portion of those aged $i-1$ who dies this year. 
# 
# Then we can model the population with the dynamic systems. 
# We consider the 0-year old first. It includes all newborn from all ages, so we have
# $$x_1(t+1) = b^\top x(t)$$
# Then we consider all other ages for $i>1$. 
# $$x_i(t+1) = (1-d_i)x_{i-1}(t)$$
# 
# 

# In[ ]:


population = np.array([
       3.94415,       3.97807,       4.09693,       4.11904,       4.06317,       4.05686,       4.06638,       4.03058,
       4.04649,       4.14835,       4.17254,       4.11442,       4.10624,       4.11801,       4.16598,       4.24282,
       4.31614,       4.39529,       4.50085,       4.58523,       4.51913,       4.35429,       4.26464,       4.19857,
       4.24936,       4.26235,       4.15231,       4.24887,       4.21525,       4.22308,       4.28567,       3.97022,
       3.98685,       3.88015,       3.83922,       3.95643,       3.80209,       3.93445,       4.12188,       4.36480,
       4.38327,       4.11498,       4.07610,       4.10511,       4.21150,       4.50887,       4.51976,       4.53526,
       4.53880,       4.60590,       4.66029,       4.46463,       4.50085,       4.38035,       4.29200,       4.25471,
       4.03751,       3.93639,       3.79493,       3.64127,       3.62113,       3.49260,       3.56318,       3.48388,
       2.65713,       2.68076,       2.63914,       2.64936,       2.32367,       2.14232,       2.04312,       1.94932,
       1.86427,       1.73696,       1.68449,       1.62008,       1.47107,       1.45533,       1.40012,       1.37119,
       1.30851,       1.21287,       1.16142,       1.07481,       0.98572,       0.91472,       0.81421,       0.71291,
       0.64062,       0.53800,       0.43556,       0.34499,       0.28139,       0.21698,       0.16944,       0.12972,
       0.09522,       0.06814,       0.04590,       0.03227])
d = np.array([
       0.00623,       0.00044,       0.00027,       0.00020,       0.00016,       0.00012,       0.00011,       0.00011,
       0.00012,       0.00011,       0.00010,       0.00013,       0.00013,       0.00015,       0.00020,       0.00025,
       0.00037,       0.00047,       0.00064,       0.00071,       0.00076,       0.00087,       0.00087,       0.00088,
       0.00094,       0.00092,       0.00095,       0.00093,       0.00099,       0.00101,       0.00103,       0.00109,
       0.00110,       0.00114,       0.00115,       0.00120,       0.00131,       0.00137,       0.00146,       0.00156,
       0.00162,       0.00185,       0.00201,       0.00216,       0.00243,       0.00258,       0.00298,       0.00325,
       0.00351,       0.00387,       0.00413,       0.00454,       0.00494,       0.00533,       0.00571,       0.00602,
       0.00670,       0.00710,       0.00769,       0.00828,       0.00860,       0.00932,       0.00998,       0.01101,
       0.01250,       0.01282,       0.01404,       0.01515,       0.01687,       0.01830,       0.01967,       0.02133,
       0.02347,       0.02562,       0.02800,       0.03083,       0.03441,       0.03711,       0.04126,       0.04448,
       0.04964,       0.05539,       0.06149,       0.06803,       0.07673,       0.08561,       0.09540,       0.10636,
       0.11802,       0.13385,       0.15250,       0.16491,       0.18738,       0.20757,       0.22688,       0.25196,
       0.27422,       0.29239,       0.32560,       0.34157])
b = np.array([
       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,
       0.00000,       0.00000,       0.00020,       0.00020,       0.00020,       0.00020,       0.00020,       0.01710,
       0.01710,       0.01710,       0.01710,       0.01710,       0.04500,       0.04500,       0.04500,       0.04500,
       0.04500,       0.05415,       0.05415,       0.05415,       0.05415,       0.05415,       0.04825,       0.04825,
       0.04825,       0.04825,       0.04825,       0.02250,       0.02250,       0.02250,       0.02250,       0.02250,
       0.00510,       0.00510,       0.00510,       0.00510,       0.00510,       0.00035,       0.00035,       0.00035,
       0.00035,       0.00035,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,
       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,
       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,
       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,
       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,
       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,       0.00000,
       0.00000,       0.00000,       0.00000,       0.00000])

plt.subplot(1,3,1)
plt.plot(population)
plt.title('population in 2010')
plt.subplot(1,3,2)
plt.plot(b)
plt.title('birth rate in 2010')
plt.subplot(1,3,3)
plt.plot(d)
plt.title('death rate in 2010')

plt.show()


# &#9989;  **<font color=red>Do this:</font>** Find the $100\times 100$ matrix `A2` such that 
# $$x(t+1)=Ax(t)$$

# In[ ]:


## Your work here


# In[ ]:


from answercheck import checkanswer
checkanswer.detailedwarnings = False
checkanswer.matrix(A,'e777eb79afb4f64da7e47f20f00fb6ec')


# Finally, we plot the distribution for our in the year 2020 in blue and our 2010 population in orange.

# In[ ]:


plt.plot(A**10*np.reshape(population,[100,1]),label = '2020 Population')
plt.plot(population,alpha=0.6,label = '2010 Population')
plt.legend(loc='lower left');


# &#9989;  **<font color=red>Do this:</font>** Describe any similarities in the two graphs.

# Your answer here

# ---
# <a name='random_walk'></a>
# ## 6. Graphs to Random Walk Markov Models
# 
# This section discusses how to use diagonalization to transform any graph into a random walk Markov chain.
# 
# The code for the section below is not overly diffiucult, but it does cover some material we haven't fully discussed. Namely the concept of a [graph in discrete mathematics](#https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)). You do not need to finish this section on your own. We can discuss it as a big group when you reach this point.
# 

# In[ ]:


# Here are some libraries you may need to use
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import sympy as sym
import networkx as nx
import matplotlib.pyplot as plt
sym.init_printing(use_unicode=True)


# ### Graph Random Walk
# 
# * Define the following matrices:
#     * $I$ is the identity matrix
#     * $A$ is the adjacency matrix
#     * $D$ is diagonal matrix of degrees (number of edges connected to each node)
#     
# $$W=\frac{1}{2}(I + AD^{-1})$$
# 
# * The **lazy random walk matrix**, $W$, takes a distribution vector of *stuff*, $p_{t}$, and diffuses it to its neighbors:
# 
# $$p_{t+1}=Wp_{t}$$
# 
# * For some initial distribution of *stuff*, $p_{0}$, we can compute how much of it would be at each node at time, $t$, by powering $W$ as follows:
# 
# $$p_{t}=W^{t}p_{0}$$
# 
# * Plugging in the above expression yields:
# 
# $$p_{t}=\left( \frac{1}{2}(I+AD^{-1}) \right)^t p_{0}$$

# **<font color=red>DO THIS</font>**: Using matrix algebra, show that $\frac{1}{2}(I + AD^{-1})$ is **similar** to  $I-\frac{1}{2}N$, where $N=D^{-\frac{1}{2}}(D-A)D^{-\frac{1}{2}}$ is the normalized graph Laplacian. 

# **Your answer goes here** (follow along after attempting)

# ### Random Walk on Barbell Graph
# 
# To generate the barbell graph, run the following cell.

# In[ ]:


n = 60 # number of nodes
B = nx.Graph() # initialize graph

## initialize empty edge lists
edge_list_complete_1 = [] 
edge_list_complete_2 = []
edge_list_path = []

## generate node lists
node_list_complete_1 = np.arange(int(n/3))
node_list_complete_2 = np.arange(int(2*n/3),n)
node_list_path = np.arange(int(n/3)-1,int(2*n/3))

## generate edge sets for barbell graph
for u in node_list_complete_1:
    for v in np.arange(u+1,int(n/3)):
        edge_list_complete_1.append((u,v))
        
for u in node_list_complete_2:
    for v in np.arange(u+1,n):
        edge_list_complete_2.append((u,v))

for u in node_list_path:
    edge_list_path.append((u,u+1))

# G.remove_edges_from([(3,0),(5,7),(0,7),(3,5)])

## add edges
B.add_edges_from(edge_list_complete_1)
B.add_edges_from(edge_list_complete_2)
B.add_edges_from(edge_list_path)


## draw graph
pos=nx.spring_layout(B) # positions for all nodes

### nodes
nx.draw_networkx_nodes(B,pos,
                       nodelist=list(node_list_complete_1),
                       node_color='c',
                       node_size=400,
                       alpha=0.8)
nx.draw_networkx_nodes(B,pos,
                       nodelist=list(node_list_path),
                       node_color='g',
                       node_size=200,
                       alpha=0.8)
nx.draw_networkx_nodes(B,pos,
                       nodelist=list(node_list_complete_2),
                       node_color='b',
                       node_size=400,
                       alpha=0.8)


### edges
nx.draw_networkx_edges(B,pos,
                       edgelist=edge_list_complete_1,
                       width=2,alpha=0.5,edge_color='c')
nx.draw_networkx_edges(B,pos,
                       edgelist=edge_list_path,
                       width=3,alpha=0.5,edge_color='g')
nx.draw_networkx_edges(B,pos,
                       edgelist=edge_list_complete_2,
                       width=2,alpha=0.5,edge_color='b')


plt.axis('off')
plt.show() # display


# &#9989;  **<font color=red>Do this</font>:** Generate the lazy random walk matrix, $W= \frac{1}{2} (I+AD^{-1})$, for the above graph. Consider the ``np.eye`` function to contruct the identity.

# In[ ]:



A = nx.adjacency_matrix(B)
A = A.todense()

d = np.sum(A,0) # Make a vector of the sums.
D = np.diag(np.asarray(d)[0])


# In[ ]:


#Put your answer to the above question here.


# In[ ]:


from answercheck import checkanswer
checkanswer.matrix(W, "7af4a5b11892da6e1a605c8239b62093")


# &#9989;  **<font color=red>Do this</font>:** Compute the eigenvalues and eigenvectors of $W$. Make a diagonal matrix $J$ with the eigenvalues on the diagonal. Name the matrix of eigenvectors $V$ (each column is an eigenvector).

# In[ ]:


#Put your answer to the above question here. 


# Now we make sure we constructed $V$ and $A$ correctly by double checking that $W = VJV^{-1}$

# In[ ]:


np.allclose(W, V*J*np.linalg.inv(V))


# &#9989;  **<font color=red>Do this</font>:** Let your $p_{0}=[1,0,0,\ldots,0]$. Compute $p_{t}$ for $t=1,2,\ldots,100$, and plot $||v_{1}-p_{t}||_{1}$ versus $t$, where $v_{1}$ is the eigenvector associated with the largest eigenvalue $\lambda_{1}=1$ and whose sum equals 1. (**Note**: $||\cdot||_{1}$ may be computed using ```np.linalg.norm(v_1-p_t, 1)```.)

# In[ ]:


#Put your answer to the above question here. 


# #### Compare to Complete Graph
# 
# If you complete the above, do the same for a complete graph on the same number of nodes.
# 
# &#9989;  **<font color=red>Question</font>:** What do you notice about the graph that is different from that above?

# Put your answer to the above question here.

# 
# 
