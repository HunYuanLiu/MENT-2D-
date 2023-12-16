# MENT-2D-
### One_Two_package.py文件中包含了从一维重构二维需要用到的函数，需要把这个文件和运行文件放在同一个文件夹中。
### Example_ Reconstructing a double ring.ipynb是一个重构的例子，其中定义了一个双重环，计算了这个双重环在12个不同方向的投影，然后用12个投影重构了双重环。
### 做重构时需要调用的函数是Two_to_one_dimensional(G,A)，它也在One_Two_package.py文件中
### 调用Two_to_one_dimensional（）函数获得重构结果; 第一个输入的参数是约束条件G，它是所有约束条件的集合，每个约束条件都是一维的曲线；第二个输入的参数是约束条件的方向A，A[i]对应G[i]的方向，A[i]是2乘2的矩阵，表述从[x,y]方向到G[i]所在的方向的映射，G[i]是二维函数通过映射A[i]后在横轴上的投影；计算出来的结果是归一化的；每轮迭代完成后都重构一个二维图像，在所有A的方向对这个二维图像积分，统计这些积分与约束条件的差值的平均值，记为differ;最终重构的结果保存在"Rho_solution.npy"
### The file One_Two_package.py contains the functions required for reconstructing two dimensions from one dimension. It is necessary to place this file and the running file in the same folder.
### Example_Reconstructing a double ring.ipynb is an example of reconstruction. It defines a double ring, calculates its projections in 12 different directions, and then reconstructs the double ring using these 12 projections.
### The function needed for reconstruction is Two_to_one_dimensional(G,A), which is also in the One_Two_package.py file.
### To obtain the reconstruction result, you can call the `Two_to_one_dimensional()` function. The function takes two input parameters:
### 1. The first parameter is the set of constraint conditions, `G`. Each constraint condition is a one-dimensional curve.
### 2. The second parameter is the direction of the constraint conditions, `A`. `A[i]` corresponds to the direction of `G[i]`. `A[i]` is a 2x2 matrix that represents the mapping from the `[x, y]` direction to the direction of `G[i]`.`G[i]` is the projection of the two-dimensional function after mapping `A[i]` onto the horizontal axis.
### The computed results are normalized. After each iteration, a two-dimensional image is reconstructed. For this image, we integrate it over all directions of A and calculate the average difference between these integrals and the constraint conditions. This average difference is denoted as "differ". 
### Save the reconstruction result in 'Rho_solution.npy'. 


### example：


```python
import matplotlib.pyplot as plt
import numpy as np
import One_Two_package as DIY
import math
import sys
#Consider the function to exist within a square of side length V.By modifying the value of V, you can control the size of the square.
V=DIY.V=16   
#Divide the square into n_points^2 small grids and represent each small grid by the coordinates of its center.
#Modify the value of n_points to adjust the pixel density of the image.
n_points=DIY.n_points=2000 
#Each small grid has a width of deltaV.
deltaV=DIY.deltaV=DIY.V/DIY.n_points 
#Modify the value of Number_constraints to indicate the number of constraints you will use in the reconstruction.
Number_constraints=DIY.Number_constraints=12 
#G and A are pieces of information that the user needs to provide and fill in.
#Please make sure to include the prefix when calling the function so that the data can be transferred to the file where the function is located.
#Both G and A have Number_constraints elements, and the positions of these elements correspond one-to-one.
DIY.G={}
DIY.A={}
DIY.Two_to_one_dimensional(DIY.G,DIY.A)
#After running, you will obtain the reconstructed two-dimensional image. 
#The corresponding two-dimensional function described in the image will be saved in matrix form as "Rho_solution.npy".
```
