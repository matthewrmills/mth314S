#!/usr/bin/env python
# coding: utf-8

# 

# # 105 Pre-Class Assignment: Determinants and Eigenvectors

# ### Readings for Determinants (Recommended in bold)
#  * [Heffron Chapter 4.I-II pg 317-337](http://joshua.smcvt.edu/linearalgebra/book.pdf)
#  * [**_Beezer Chapter D pg 340-366_**](http://linear.ups.edu/download/fcla-3.50-tablet.pdf)
# 
# ### Readings for this Eigen-stuff (Recommended in bold)
#  * [Heffron Chapter  5 II.3 pg 397-407](http://joshua.smcvt.edu/linearalgebra/book.pdf)
#  * [Beezer Chapter E pg 367-369](http://linear.ups.edu/download/fcla-3.50-tablet.pdf)
# 
# 

# ### Goals for today's pre-class assignment 
# 
# 
# 1. [Introduction to Determinants](#Introduction_to_Determinants)
# 1. [Properties of Determinants](#Properties_of_Determinants)
# 1. [One interpretation of determinants](#One_Interpretation_of_determinants)
# 1. [Cramer's Rule](#CramersRule)
# 1. [Eigenvectors and Eigenvalues](#Eigenvectors_and_Eigenvalues)
# 2. [Solving Eigenproblems - A 2x2 Example](#Solving_Eigenproblems)
# 3. [Introduction to Markov Models](#Markov_Models)
# 1. [Assignment wrap-up](#T3)

# ----
# 
# <a name="Introduction_to_Determinants"></a>
# ## 1. Introduction to Determinants
# 
# For a detailed overview of determinants I would recommend reviewing **_Chapter D pg 340-366_** of the Beezer text.  
# 
# The determinant is a function that takes a ($n \times n$) square matrix as an input and produces a scalar as an output. Determinants have been studied quite extensively and have many interesting properties.  However, determinants are "computationally expensive" as the size of your matrix ($n$) gets bigger.  This limitation makes them impractical for many real world problems.  
# 
# The determinant of a $ 2 \times 2$ matrix can be calculated as follows:
# 
# $$ 
# det \left(
# \left[
# \begin{matrix}
#     a_{11} & a_{12}  \\
#     a_{21} & a_{22}
# \end{matrix}
# \right] 
# \right)
# = a_{11}a_{22} - a_{12}a_{21}
# $$

# 
# &#9989; **<font color=red>QUESTION:</font>** Calculate the determinant of the following matrix by hand:
# 
# $$ 
# \left[
# \begin{matrix}
#     3 & -2  \\
#     1 & 2
# \end{matrix}
# \right] 
# $$

# In[1]:


#Put your answer here


# Calculating the determinant of a larger matrix is a "recursive" problem which involves combining the determinants of smaller and smaller sub-matrices until you have a $2 \times 2$ matrix which is then calculated using the above formula.  Here is some Pseudocode to calculate a determinant.  To simplify the example the code assumes there is a matrix function ```deleterow``` which will remove the $x$th row from a matrix  (always the first row in this example) and ```deletecol``` will remove the $x$th column from a matrix. When used together (as shown below) they will take an $n \times n$ matrix and turn it into a $ (n-1) \times (n-1)$ matrix.  
# 
# 
# ```bash
# function determinant(A, n)
#    det = 0
#    if (n == 1)
#       det = matrix[1,1]
#    else if (n == 2)
#       det = matrix[1,1] * matrix[2,2] - matrix[1,2] * matrix[2,1]
#    else 
#       for x from 1 to n
#           submatrix = deleterow(matrix, 1)
#           submatrix = deletecol(submatrix, x)
#           det = det + (x+1)**(-1) * matrix[1,x] * determinant(submatrix, n-1)
#       next x
#    endif
#    
#    return det
# ```

# Notice that the combination of the determinants of the submatrixes is not a simple sum.  The combination is adding the submatrices corresponding to the odd columns (1,3,5, etc) and subtracting the submatrices corresponding to the even columns (2,4,6, etc.). This may become clearer if we look at a simple $3 \times 3$ example (Let $|A|$ be a simplified syntax for writing the determinant of $A$):
# 
# $$
# A = \left[
# \begin{matrix}
#     a_{11} & a_{12} & a_{13} \\
#     a_{21} & a_{22} & a_{23} \\
#     a_{31} & a_{32} & a_{33} 
# \end{matrix}
# \right] $$
# 
# $$ 
# |A|=
# a_{11} \left|
# \begin{matrix}
#     \square & \square  & \square  \\
#     \square  & a_{22} & a_{23} \\
#     \square  & a_{32} & a_{33} 
# \end{matrix}
# \right|
# -
# a_{12}\left|
# \begin{matrix}
#     \square & \square  & \square  \\
#     a_{21} & \square & a_{23} \\
#     a_{31} & \square & a_{33} 
# \end{matrix}
# \right|
# +
# a_{13} \left|
# \begin{matrix}
#     \square & \square  & \square  \\
#     a_{21} & a_{22} & \square \\
#     a_{31} & a_{32} & \square
# \end{matrix}
# \right|
# $$
# 
# 
# $$ 
# |A|
# =
# a_{11}\left|
# \begin{matrix}
#     a_{22} & a_{23}  \\
#     a_{32} & a_{33}
# \end{matrix}
# \right|
# -
# a_{12}\left|
# \begin{matrix}
#     a_{21} & a_{23}  \\
#     a_{31} & a_{33}
# \end{matrix}
# \right|
# +
# a_{13}
# \left|
# \begin{matrix}
#     a_{21} & a_{22}  \\
#     a_{31} & a_{32}
# \end{matrix}
# \right|
# $$
# 
# $$
# |A| = 
# a_{11}(a_{22}a_{33} - a_{23}a_{32})
# -
# a_{12}(a_{21}a_{33} - a_{23}a_{31})
# +
# a_{13}(a_{21}a_{32} - a_{22}a_{31})
# $$

# &#9989; **<font color=red>QUESTION:</font>** Calculate the determinant of the following matrix by hand:
# 
# $$ 
# \left[
# \begin{matrix}
#     1 & 2 & -3  \\
#     5 & 0 & 6  \\
#     7 & 1 & -4
# \end{matrix}
# \right] 
# $$

# Put your answer here

# &#9989; **<font color=red>QUESTION:</font>** Import ``numpy`` and use the ```numpy.linalg``` library to calculate the determinant of the following matrix and store the value in a variable called ```det```
# 
# $$
# \left[
# \begin{matrix}
#     2 & 0 & 1 & -5  \\
#     8 & -1 & 2 & 1  \\
#     4 & -3 & -5 & 0 \\
#     1 & 4 & 8 & 2
# \end{matrix}
# \right] 
# $$

# In[2]:


#Put your answer here


# In[3]:


from answercheck import checkanswer

checkanswer.float(det,'49afb719e0cd46f74578ebf335290f81');


# ----
# 
# <a name="Properties_of_Determinants"></a>
# ## 2. Properties of Determinants
# 
# The following are some helpful properties when working with determinants.  These properties are often used in proofs and can sometimes be utilized to make faster calculations.
# 
# ### Row Operations
# 
# Let $A$ be an $n \times n$ matrix and $c$ be a nonzero scalar. Let $|A|$ be a simplified syntax for writing the determinant of $A$: 
# 
# 1. If a matrix $B$ is obtained from $A$ by multiplying a row (column) by $c$ then $|B| = c|A|$.
# 2. If a matrix $B$ is obtained from $A$ by interchanging two rows (columns) then $|B| = -|A|$.
# 3. if a matrix $B$ is obtained from $A$ by adding a multiple of one row (column) to another row (column), then $|B| = |A|$.
# 
# 

# ### Singular Matrices
# 
# **Definition:** A square matrix $A$ is said to be **singular** if $|A| = 0$. $A$ is **nonsingular** if $|A| \neq 0$
# 
# Now, Let $A$ be an $n \times n$ matrix. $A$ is singular if any of these is true:
# 
# 1. all the elements of a row (column) are zero.
# 2. two rows (columns) are equal.
# 3. two rows (columns) are proportional. i.e. one row (column) is the same as another row (column) multiplied by $c$.
# 4. one row (column) is a linear combination of the others.
# 

# &#9989; **<font color=red>QUESTION:</font>** The following matrix is singular because of certain column or row properties. Give the reason:
# 
# $$ 
# \left[
# \begin{matrix}
#     1 & 5 & 5  \\
#     0 & -2 & -2  \\
#     3 & 1 & 1
# \end{matrix}
# \right] 
# $$

# Put your answer here.

# &#9989; **<font color=red>QUESTION:</font>** The following matrix is singular because of certain column or row properties. Give the reason:
# 
# $$ 
# \left[
# \begin{matrix}
#     1 & 0 & 4  \\
#     0 & 1 & 9  \\
#     0 & 0 & 0
# \end{matrix}
# \right] 
# $$

# Put your answer here.

# ### Determinants and Matrix Operations
# 
# Let $A$ and $B$ be $n\times n$ matrices and $c$ be a nonzero scalar.
# 
# 1. Determinant of a scalar multiple: $|cA| = c^n|A|$
# 2. Determinant of a product: $|AB| = |A||B|$
# 3. Determinant of a transpose" $|A^t| = |A|$
# 4. Determinant of an inverse: $|A^{-1}| = \frac{1}{|A|}$ (Assuming $A^{-1}$ exists)

# &#9989; **<font color=red>QUESTION:</font>**  If $A$ is a $3\times 3$ matrix with $|A| = 3$, use the properties of determinants to compute the following determinant:
# 
# $$|2A|$$

# Put your answer here

# &#9989; **<font color=red>QUESTION:</font>**  If $A$ is a $3\times 3$ matrix with $|A| = 3$, use the properties of determinants to compute the following determinant:
# $$|A^2|$$

# Put your answer here

# &#9989; **<font color=red>QUESTION:</font>**  if $A$ and $B$ are $3\times 3$ matrices and $|A| = -3, |B|=2$, compute the following determinant:
# 
# $$|AB|$$
# 

# Put your answer here

# &#9989; **<font color=red>QUESTION:</font>**  if $A$ and $B$ are $3\times 3$ matrices and $|A| = -3, |B|=2$, compute the following determinant:
# 
# $$|2AB^{-1}|$$

# Put your answer here

# ### Triangular matrices
# 
# **Definition:** A matrix is said to be **upper triangular** if all nonzero elements lie on or above the main diagonal and all elements below the main diagonal are zero. For example:
# 
# 
# $$ A = 
# \left[
# \begin{matrix}
#     2 & -1 & 9 & 4  \\
#     0 & 3 & 0 & 6 \\
#     0 & 0 & -5 & 3 \\
#     0 & 0 & 0 & 1
# \end{matrix}
# \right] 
# $$
# 
# **Definition:** A matrix is said to be **lower triangular** if its transpose is upper triangular.
# 
# The determinant of an *upper triangle matrix* $A$ is the product of the diagonal elements of the matrix $A$.  
# 
# Also, since the Determinant is the same for a matrix and it's transpose (i.e.  $|A^\top| = |A|$, see definition above) the determinant of a *lower triangle matrix* is also the product of the diagonal elements. 

# &#9989; **<font color=red>QUESTION:</font>**   What is the determinant of matrix $A$?

# Put your answer here

# ### Using Properties of determinants:
# Here is a great video showing how you can use the properties of determinants:

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("aKX5_DucNq8",width=640,height=360, cc_load_policy=True)


# &#9989; **<font color=red>QUESTION (A challenging one):</font>**   Using the pattern established in the video can you calculate the determinate of the following matrix?
# 
# $$ 
# \left[
# \begin{matrix}
#     1 & a & a^2 & a^3 \\
#     1 & b & b^2 & b^3 \\
#     1 & c & c^2 & c^3 \\
#     1 & d & d^2 & d^3 
# \end{matrix}
# \right] 
# $$
# 
# **Note.** You can also attempt to do this calculation in sympy.

# Put your answer here

# ----
# 
# <a name="One_Interpretation_of_determinants"></a>
# ## 3. One interpretation of determinants
# 
# The following is an application of determinants. Watch this!

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("Ip3X9LOh2dk",width=640,height=360, cc_load_policy=True)


# For fun, we will recreate some of the video's visualizations in Python. 
# It was a little tricky to get the aspect ratios correct but here is some code I managed to get it work. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as npimport sympy as sym


# In[ ]:


# Lets define somme points that form a Unit Cube
points = np.array([[0, 0, 0],
                  [1, 0, 0 ],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1 ],
                  [1, 1, 1],
                  [0, 1, 1]])

points = np.matrix(points)


# In[ ]:


#Here is some code to build cube from https://stackoverflow.com/questions/44881885/python-draw-3d-cube

def plot3dcube(Z):
    
    if type(Z) == np.matrix:
        Z = np.asarray(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    r = [-1,1]

    X, Y = np.meshgrid(r, r)
    # plot vertices
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

    # list of sides' polygons of figure
    verts = [[Z[0],Z[1],Z[2],Z[3]],
     [Z[4],Z[5],Z[6],Z[7]], 
     [Z[0],Z[1],Z[5],Z[4]], 
     [Z[2],Z[3],Z[7],Z[6]], 
     [Z[1],Z[2],Z[6],Z[5]],
     [Z[4],Z[7],Z[3],Z[0]], 
     [Z[2],Z[3],Z[7],Z[6]]]

    #alpha transparency was't working found fix here: 
    # https://stackoverflow.com/questions/23403293/3d-surface-not-transparent-inspite-of-setting-alpha
    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, 
     facecolors=(0,0,1,0.25), linewidths=1, edgecolors='r'))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ## Weird trick to get the axpect ratio to work.
    ## From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    mx = np.amax(Z, axis=0)
    mn = np.amin(Z, axis=0)
    max_range = mx-mn

    # Create cubic bounding box to simulate equal aspect ratio
    Xb = 0.5*max_range.max()*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(max_range[0])
    Yb = 0.5*max_range.max()*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(max_range[1])
    Zb = 0.5*max_range.max()*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(max_range[2])
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()


# In[ ]:


plot3dcube(points)


# **<font color='red'>QUESTION:</font>** The following the $3\times 3$ was shown in the video (around 6'50''). Apply this matrix to the unit cube and use the ```plot3dcube``` to show the resulting transformed points. 
# 
# 

# In[ ]:


T = np.matrix([[1 , 0 ,  0.5],
               [0.5 ,1 ,1.5],
               [1 , 0 ,  1]])

#Put the answer to the above question here. 


# &#9989; **<font color='red'>QUESTION:</font>** The determinant represents how the area changes when applying a $2 \times 2$ transform.  What does the determinant represent for a $3 \times 3$ transform?

# Put your answer here

# ----
# 
# <a name="CramersRule"></a>
# ## 4. Cramer's Rule
# 
# &#9989; **<font color='red'>DO THIS:</font>** Watch the following video and come to class ready to discuss Cramer's Rule. We will implement this method to solve systems of equations tomorrow in class.

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("BW6897HIOMA",width=640,height=360, cc_load_policy=True)


# ----
# 
# <a name="Eigenvectors_and_Eigenvalues"></a>
# 
# The other application of deteriminants (which we will use almost everyday the rest of the semester) is eigenvectors.
# 
# ## 5. Eigenvectors and Eigenvalues
# 
# Understanding Eigenvector and Eigenvalues can be very challenging. These are complex topics with many facets.  Different textbooks approach the problem from different directions.  All have value.  These facets include:
# 
# - Understanding the mathematical definition of Eigenvalues.
# - Being able to calculate an Eigenvalue and Eigenvector.
# - Understanding what Eigenvalues and Eigenvectors represent. 
# - Understanding how to use Eigenvalues and Eigenvectors to solve problems. 
# 
# In this course we consider it more important to understand what eigenvectors and eigenvalues represent and how to use them. However, often this understanding comes from first learning how to calculate them.  
# 
# > Eigenvalues are a special set of scalars associated with a **square matrix** that are sometimes also known as characteristic roots, characteristic values (Hoffman and Kunze 1971), proper values, or latent roots (Marcus and Minc 1988, p. 144).
# 
# > The determination of the eigenvalues and eigenvectors of a matrix is extremely important in physics and engineering, where it is equivalent to matrix diagonalization and arises in such common applications as [stability analysis](https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis), the [physics of rotating bodies](http://www.physics.princeton.edu/~mcdonald/examples/ph101_1996/ph101lab5_96.pdf), and [small oscillations of vibrating systems](http://lpsa.swarthmore.edu/MtrxVibe/MatrixAll.html), to name only a few.
# 
# > The decomposition of a square matrix $A$ into eigenvalues and eigenvectors is known in this work as eigen decomposition, and the fact that this decomposition is always possible as long as the matrix consisting of the eigenvectors of $A$ is square. This is known as the eigen decomposition theorem.
# 
# 
# From: http://mathworld.wolfram.com/Eigenvalue.html

# The following video provides an intuition for eigenvalues and eigenvectors.  

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("ue3yoeZvt8E",width=640,height=360, cc_load_policy=True)


# ### Definition
# 
# Let $A$ be an $n\times n$ matrix.  An eigenvector of $A$ ass is a non-zero vector $x$ in $R^n$ such that:
# 
# $$Ax=\lambda x$$
# 
# for some scalar $\lambda \in \mathbb{R}$. (The values of $\lambda$ and $x$ can be complex numbers in general, but we will only focus on real values in this course). The scalar $\lambda$ above is called the **eigenvalue** associated to $x$.
# 
# In order to compute these eigenvalues/eigenvectors we begin by rewritting the equation as $Ax=\lambda x$ as:
# 
# $$(A-\lambda I_n)x = 0$$
# 
# Naturally, there is the trivial solution is $x=0$, but recall we defined eigenvectors to be nonzero. 
# 
# Now, nonzero (i.e. non-trivial) solutions to the system of equations $(A-\lambda I_n)x = 0$ can only exist if the matrix $(A-\lambda I_n)$ is singular, i.e. the determinant of $|A - \lambda I_n| = 0$. (This comes from the Invertible Matrix Theorem, but we will talk about it more later.)
# 
# Therefore, solving the equation $|A - \lambda I_n| = 0$ for $\lambda$ leads to all the eigenvalues of $A$.
# 
# **Note:** the above logic is key.  Make sure you understand. If not, ask questions. 
# 
# The following video gives a geometric description of eigenvalues and eigenvectors. In terms of affine transformations, the eigenvectors of $A$ are the vectors in which the matrix $A$ will always act as a scaling matrix. 

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("PFDu9oVAE-g",width=640,height=360, cc_load_policy=True)


# ### Examples:
# Here are a few more examples of how eigenvalues and eigenvectors are used (You are not required to understand all, but they give examples of things we could use as a homework topic later in the semester if there is interest.):
# 
# > [Using singular value decomposition for image compression](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxuYXNsdW5kZXJpY3xneDpkMTI4OTI1NTc4YjRlOGE). 
# This is a note explaining how you can compress an image by throwing away the small eigenvalues of $A^TA$. 
# It takes an 88 megapixel image of an Allosaurus and shows how the image looks after compressing by selecting the largest singular values.
# 
# > [Deriving Special Relativity]((https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxuYXNsdW5kZXJpY3xneDo2ZTAyNzA4NTZmOGZmNmU4) is more natural in the language of linear algebra. 
# In fact, Einstein's second postulate really states that "Light is an eigenvector of the Lorentz transform." 
# This document goes over the full derivation in detail.
# 
# > [Spectral Clustering](https://en.wikipedia.org/wiki/Spectral_clustering). 
# Whether it's in plants and biology, medical imaging, buisness and marketing, understanding the connections between fields on Facebook, or even criminology, clustering is an extremely important part of modern data analysis. 
# It allows people to find important subsystems or patterns inside noisy data sets. 
# One such method is spectral clustering, which uses the eigenvalues of the graph of a network. 
# Even the eigenvector of the second smallest eigenvalue of the Laplacian matrix allows us to find the two largest clusters in a network.
# 
# > [Dimensionality Reduction/PCA](https://en.wikipedia.org/wiki/Principal_component_analysis). 
# The principal components correspond to the largest eigenvalues of $A^\top A$, and this yields the least squared projection onto a smaller dimensional hyperplane, and the eigenvectors become the axes of the hyperplane. 
# Dimensionality reduction is extremely useful in machine learning and data analysis as it allows one to understand where most of the variation in the data comes from.
# 
# > [Low rank factorization for collaborative prediction](http://cs229.stanford.edu/proj2006/KleemanDenuitHenderson-MatrixFactorizationForCollaborativePrediction.pdf). 
# This is what Netflix does (or once did) to predict what rating you'll have for a movie you have not yet watched. 
# It uses the singular value decomposition and throws away the smallest eigenvalues of $A^\top A$.
# 
# > [The Google Page Rank algorithm](https://en.wikipedia.org/wiki/PageRank). 
# The largest eigenvector of the graph of the internet is how the pages are ranked.
# 
# From: https://math.stackexchange.com/questions/1520832/real-life-examples-for-eigenvalues-eigenvectors

# ----
# <a name="Solving_Eigenproblems"></a>
# ## 6. Solving Eigenproblems - A 2x2 Example
# 
# We will do a basic example of computing eigenvalues here, and get more involved in class tomorrow.

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("0UbkMlTu1vo",width=640,height=360, cc_load_policy=True)


# Consider calculating eigenvalues for any $2\times 2$ matrix. 
# We want to solve:
# 
# $$|A - \lambda I_2 | = 0$$
# 
# $$ 
# \left|
# \left[
# \begin{matrix}
#     a_{11} & a_{12} \\
#     a_{21} & a_{22}
# \end{matrix}
# \right] 
# - \lambda \left[
# \begin{matrix}
#     1 & 0 \\
#     0 & 1
# \end{matrix}
# \right] 
# \right|
# =
# \left|
# \left[
# \begin{matrix}
#     a_{11}-\lambda & a_{12} \\
#     a_{21} & a_{22}-\lambda
# \end{matrix}
# \right]
# \right|
# =0
# $$
# 
# We know this determinant:
# 
# $$(a_{11}-\lambda)(a_{22}-\lambda) - a_{12} a_{21}  = 0 $$
# 
# If we expand the above, we get:
# 
# $$a_{11}a_{22}+\lambda^2-a_{11}\lambda-a_{22}\lambda - a_{12} a_{21} = 0$$
# 
# and
# 
# $$\lambda^2-(a_{11}+a_{22})\lambda+a_{11}a_{22} - a_{12} a_{21} = 0$$
# 
# 
# This is a simple quadratic equation. 
# The roots pf $A\lambda^2+B\lambda+C = 0$ can be solved using the quadratic formula:
# 
# $$ \frac{-B \pm \sqrt{B^2 - 4AC}}{2A}$$
# 
# &#9989; **<font color=red>QUESTION:</font>** Using the above equation.  What are the eigenvalues for the following $2\times 2$ matrix. Try calculating this by hand and then store the lower value in a variable named```e1``` and the larger value in ```e2``` to check your answer:
# 
# $$A =
# \left[
# \begin{matrix}
#     -4 & -6  \\
#     3 & 5
#  \end{matrix}
# \right] 
# $$

# In[ ]:


# Put your answer here


# In[ ]:


from answercheck import checkanswer

checkanswer.float(e1,'c54490d3480079138c8c027a87a366e3');


# In[ ]:


from answercheck import checkanswer

checkanswer.float(e2,'d1bd83a33f1a841ab7fda32449746cc4');


# &#9989; **<font color=red>DO THIS</font>** Find a ```numpy``` function that will calculate eigenvalues and verify the answers from above.

# In[ ]:


# Put your answer here


# &#9989; **<font color=red>QUESTION:</font>** What are the corresponding eigenvectors to the matrix $A$? This time you can try calculating by hand or just used the function you found in the previous answer.  Store the eigenvector associated with the ```e1``` value in a vector named ```v1``` and the eigenvector associated with the eigenvalue ```e2``` in a vector named ```v2``` to check your answer.  

# Put your answer to the above question here.

# In[ ]:


from answercheck import checkanswer

checkanswer.eq_vector(v1,'35758bc2fa8ff4f04cfbcd019844f93d');


# In[ ]:


from answercheck import checkanswer

checkanswer.eq_vector(v2,'90b0437e86d2cf70050d1d6081d942f4');


# ---
# <a name="T3"></a>
# ## 7. Assignment wrap up
# 
# 
# Please fill out the form that appears when you run the code below.  **You must completely fill this out in order to receive credit for the assignment!** If you cannont load the form below please try logging in to [spartan365.msu.edu](http://spartan365.msu.edu/) and try running it again, or simply use the direct link provided below. 
# 
# [Direct Link to Microsoft Form](https://forms.office.com/r/n0PEF9xt59)
# 

# &#9989; **<font color=red>Assignment-Specific QUESTION:</font>** What does the determinant represent for a $3 \times 3$ transform?

# Put your answer to the above question here

# &#9989; **<font color=red>Assignment-Specific QUESTION:</font>** Both **sympy** and **numpy** can calculate many of the same things. What is the fundamental difference between these two libraries?

# In[ ]:


Put your answer to the above question here.


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
# ###EndPreClass###`

# ---
# Written by Dr. Dirk Colbry, and Dr. Matthew Mills, Michigan State University
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
# 
# ----

# 
# 
