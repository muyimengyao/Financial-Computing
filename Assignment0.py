#!/usr/bin/env python
# coding: utf-8

# # 1 Set Up Your Computational Tools
# ## 1.1 Git
# 
# 
# This is my
#     [github repository](https://github.com/muyimengyao/Financial-Computing)
# 

# ## 1.2 Basic Python
# 1. Simulate 3 series of normal gaussian random variables (textithint: use numpy.random.normal) each with mean 0.05 and standard deviation 1, and store them in 3 variables named r1, r2 andr3. There series represent rate of return
# 

# In[17]:


import numpy as np
mu = 0.05
sigma = 1 
np.random.seed(0)
r1 = np.random.normal(mu, sigma, 100 )
r2 = np.random.normal(mu, sigma, 100 )
r3 = np.random.normal(mu, sigma, 100 )


# 2. Calculate the average and the standard deviation of each of these 3 series
# 

# In[18]:


print(r1.mean(),r1.std())
print(r2.mean(),r2.std())
print(r3.mean(),r3.std())


# 3. Print an alert message if the average of any of the 3 series is negative
# 

# In[29]:


alert = False
for i in [r1,r2,r3]:
    if i.mean()<0:
        alert = True
if alert:
    print("There exists a series that whose average is negative.")


# 4. Wrap steps 1-3 in a function that takes arguments $n$ (number of series), $\mu$ (mean of normal) and $σ^{2}$ (variance of normal) and return the alert message

# In[49]:


def alert_neg(n, mu, sig_2):
    r1 = np.random.normal(mu, np.sqrt(sig_2), n)
    r2 = np.random.normal(mu, np.sqrt(sig_2), n)
    r3 = np.random.normal(mu, np.sqrt(sig_2), n)
    print(r1.mean(),r2.mean(),r3.mean())
    alert = False
    for i in [r1,r2,r3]:
        if i.mean()<0:
            alert = True
    if alert:
        print("There exists a series that whose average is negative.")


# In[50]:


alert_neg(100,0.05,1)


# # 2 Numerical Computing
# ## 2.1 Newton’s Method
# Write a numerical routine that find the root of the equation:
# $$f = (1+0.5x)^{5} = 1.5$$
# 
# 1. Define this function as python function

# In[51]:


def myfunc(x):
    y = (1+0.5*x)**5
    return y


# 2. Define a function for numerical gradient:

# In[61]:


def numerical_grad(func, x, dx = 1e-5):
    dy = func(x+dx)-func(x)
    ngrad = dy/dx
    return ngrad


# 3. Define a function for Newton method with

# In[78]:


def newton_method(func , func_value = 1.5 , x = -1 , max_iteration =int(1e6) , max_err = 1e-6 ):
    for trial in range (max_iteration):
        error = func ( x ) - func_value
        if abs ( error ) < max_err:
            y = func(x)
            print ( " Iteration {}: f = {} , x = {} " . format ( trial , y ,x))
            return x
        else:
            grad = numerical_grad(func,x)
            x = x - error/grad
            y = func(x)
            print ( " Iteration {}: f = {} , x = {} " . format ( trial , y , x ))
    raise ValueError ( " Max iteration reached " )
    print ( " Max Iteration {}: f = {} , x = {} " . format ( trial , y , x ) )


# In[79]:


newton_method(myfunc)


# ## 2.2 Gradient Descent
# Implement the gradient descent to find the minimum point of the function
# $$f = x^{2}-2x+3$$
# 1. Write $f$ as a function in python

# In[80]:


def myfunc1(x):
    y = x**2-2*x+3
    return y


# 2. Write a function for the numerical gradient

# In[81]:


def numerical_grad(func, x, dx = 1e-5):
    dy = func(x+dx)-func(x)
    ngrad = dy/dx
    return ngrad


# 3. Using gradient descent:
#     - Start with an initial guess for x
#     - Update the guess as x’ = x - d * grad , where d = 0.01 is a learning rate
#     - Stop when abs(df)<0.000001, where df = f(x’) - f(x)

# In[92]:


def gradient_descent(func , x = 0 ,d = 0.01, max_iteration =int(1e6) , max_err = 1e-6 ):
    for trial in range (max_iteration):
        df = d*numerical_grad(func,x)# = f(x')-f(x)
        print(df)
        if abs ( df ) < max_err:
            y = func(x)
            print ( " Iteration {}: f = {} , x = {} " . format ( trial , y ,x))
            return x
        else:
            grad = numerical_grad(func,x)
            x = x - d*grad
            y = func(x)
            print ( " Iteration {}: f = {} , x = {} " . format ( trial , y , x ))
    raise ValueError ( " Max iteration reached " )
    print ( " Max Iteration {}: f = {} , x = {} " . format ( trial , y , x ) )


# In[94]:


gradient_descent(myfunc1)

