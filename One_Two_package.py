#!/usr/bin/env python
# coding: utf-8

# ### 这是一个自定义的函数包，包含从一维到二维的重构可能需要的python函数
# ### This is a custom function package.Includes Python functions that may be needed for reconstruction from one-dimensional to two-dimensional.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys




# ### Draw(function,direction_1_name,direction_2_name):
# ### 调用这个函数绘制二维函数function，direction_1_name是横轴的名字，direction_2_name是纵轴的名字；direction_1_name和direction_2_name都请输入文本；function是一个矩阵形式，第一个维度是横轴，第二个维度是纵轴；
# ### Call this function to plot the two-dimensional function 'function'. Please enter text for 'direction_1_name' as the name of the horizontal axis and 'direction_2_name' as the name of the vertical axis. 'function' is in matrix form, with the first dimension representing the horizontal axis and the second dimension representing the vertical axis.

# In[ ]:


def Draw(function,direction_1_name,direction_2_name):
    Function=np.transpose(function)
    tnam=direction_2_name
    direction_2_name=direction_1_name
    direction_1_name=tnam
    i = np.sum(Function)
    if (i!=0):
        Function = Function / i
    plt.imshow(Function, cmap='rainbow', extent=(-V/2, V/2, -V/2, V/2),origin='lower')
    plt.xlabel(direction_2_name)
    plt.ylabel(direction_1_name, rotation=0)
    plt.colorbar()
    plt.show()


# ### rotation_matrix_2d（theta）：
# ### 这个函数可以定义一个旋转映射，输入参数theta；该矩阵使二维函数映射到比原坐标系的逆时针旋转theta角度的新坐标系
# ### This function defines a rotation mapping with an input parameter theta. The matrix associated with this function maps a two-dimensional function to a new coordinate system that is rotated counterclockwise by an angle of theta compared to the original coordinate system.

# In[ ]:


def rotation_matrix_2d(theta):
    """
    定义一个二维旋转矩阵
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # 定义旋转矩阵
    rotation_matrix = np.array([[cos_theta , sin_theta],
                                [-sin_theta, cos_theta]])
    return rotation_matrix


# ### constant_1d_function(u,h)：
# ### 这个函数对定义在网格中的一维函数h做一阶线性近似处理，两个输入参数分别是坐标u，一维函数h，返回h在u点的值
# ### This function performs a first-order linear approximation on a one-dimensional function h defined on a grid. It takes two input parameters, the coordinate u and the one-dimensional function h, and returns the value of h at the point u

# In[ ]:


def constant_1d_function(u,h):
    global V
    global n_points
    global deltaV
    index_u=math.floor(((u +V/2)/deltaV))
    coordinate_index=(-V/2+deltaV/2)+index_u*deltaV
    O_1=0
    O_2=0
    if (u<coordinate_index):
        index_near=index_u-1
    if (u>coordinate_index):
        index_near=index_u+1
    if (u==coordinate_index):
        index_near=index_u
    if (0<=index_u<n_points):
        O_1=h[index_u]
    if (0<=index_near<n_points):
        O_2=h[index_near]
    
    L=abs(u-coordinate_index)
    value=(O_2-O_1)*(L/deltaV)+O_1
    return value

# ### constant_function(u,h)：
# ### 这个函数对定义在网格中的二维函数h做一阶线性近似处理，三个输入参数分别是坐标(u_N,u_N_)，二维函数h，返回h在(u_N,u_N_)点的值
# ### This function performs a first-order linear approximation on a two-dimensional function h defined on a grid. It takes three input parameters: coordinates (u_N, u_N_), the two-dimensional function h, and returns the value of h at the point (u_N, u_N_).

# In[ ]:

def constant_function(u_N,u_N_,h):
    #用距离所输入的坐标最近的4个网格的中心点的值为函数赋予在该坐标的值
    global V
    global deltaV
    index_u_N =math.floor(((u_N +V/2)/deltaV))
    index_u_N_=math.floor((u_N_+V/2)/deltaV)
    coordinate_index_u_N =(-V/2+deltaV/2)+index_u_N *deltaV
    coordinate_index_u_N_=(-V/2+deltaV/2)+index_u_N_*deltaV
    if(u_N>coordinate_index_u_N):
        index_u_N_near=index_u_N+1
    if(u_N==coordinate_index_u_N):
        index_u_N_near=index_u_N
    if(u_N<coordinate_index_u_N):
        index_u_N_near=index_u_N-1
    
    if(u_N_>coordinate_index_u_N_):
        index_u_N__near=index_u_N_+1
    if(u_N_==coordinate_index_u_N_):
        index_u_N__near=index_u_N_
    if(u_N_<coordinate_index_u_N_):
        index_u_N__near=index_u_N_-1
    
    if (0<=index_u_N<=n_points-1 and 0<=index_u_N_<=n_points-1):
        O_1=h[index_u_N][index_u_N_]
    else:
        O_1=0
    
    if (0<=index_u_N_near<=n_points-1 and 0<=index_u_N_<=n_points-1):
        O_2=h[index_u_N_near][index_u_N_]
    else:
        O_2=0
        
    if (0<=index_u_N<=n_points-1 and 0<=index_u_N__near<=n_points-1):
        O_3=h[index_u_N][index_u_N__near]
    else:
        O_3=0
        
    if (0<=index_u_N_near<=n_points-1 and 0<=index_u_N__near<=n_points-1):
        O_4=h[index_u_N_near][index_u_N__near]
    else:
        O_4=0

    a=abs(u_N-coordinate_index_u_N)
    b=abs(u_N_-coordinate_index_u_N_)
    
    value=(b/deltaV)*((a/deltaV)*O_4+((deltaV-a)/deltaV)*O_3)+((deltaV-b)/deltaV)*((a/deltaV)*O_2+((deltaV-a)/deltaV)*O_1)

    return value

# ### rotate(u_n,A,h_N):
# ### 这是一个坐标变化的函数，用于把在第N个坐标系的横轴上的一维函数映射到到第n个坐标系的纵轴上，在做这个映射时认为第n个坐标系的横轴坐标为常数u_n；
# ### 其中3个输入参数分别是：u_n第n个坐标系的横轴坐标，A是第n个坐标系到第N个坐标系的映射矩阵，h_N是第N个坐标系的横轴上的一维函数
# ### This is a coordinate transformation function that maps a one-dimensional function on the horizontal axis of the Nth coordinate system to the vertical axis of the nth coordinate system. During this mapping, the horizontal axis coordinate of the nth coordinate system is considered as a constant u_n.
# ### There are three input parameters:
# ### u_n: The horizontal axis coordinate of the nth coordinate system.
# ### A: The mapping matrix from the nth coordinate system to the Nth coordinate system.
# ### h_N: The one-dimensional function on the horizontal axis of the Nth coordinate system.

# In[ ]:


def rotate(u_n,A,h_N):
    global V
    global n_points
    global deltaV
    matrix=np.zeros(n_points)
    for index_v in range(n_points):
        v_n=(-V/2+deltaV/2)+index_v*deltaV
        Target=[u_n,v_n]
        u_N=np.dot(A[0],Target)
        matrix[index_v]=constant_1d_function(u_N, h_N)
    return matrix


# ### Line(function,direction)：
# ### 对二维函数function积分，我习惯认为direction为0时是对横轴积分，direction为1时是对纵轴积分
# ### When integrating the two-dimensional function 'function', I have the convention that setting 'direction' to 0 represents integrating along the horizontal axis, and setting 'direction' to 1 represents integrating along the vertical axis.

# In[ ]:


def Line(function,direction):
    global n_points
    if (direction==0):
        integral=np.sum(function,axis=0)
    
    if (direction==1):
        integral=np.sum(function,axis=1)
    


    return integral


# ### Transmit_2d(matrix,rho)：
# ### 使二维函数rho通过传输矩阵matrix，返回新的二维函数
# ### Transform the two-dimensional function rho using the transfer matrix matrix to obtain a new two-dimensional function.

# In[ ]:


def Transmit_2d(matrix,rho):
    global V
    global n_points
    global deltaV
    new_rho=np.zeros((n_points,n_points))
    matrix=np.linalg.inv(matrix)
    for index_u in range(n_points):
        for index_v in range(n_points):
            u=-V/2+deltaV/2+index_u*deltaV
            v=-V/2+deltaV/2+index_v*deltaV
            Target=[u,v]
            Source=np.dot(matrix,Target)
            x=Source[0]
            y=Source[1]
            new_rho[index_u][index_v]=constant_function(x, y, rho)
    return new_rho


# ### Solution_2d():调用这个函数用已有的h函数计算二维分布
# ### Call this function to calculate a two-dimensional distribution using an existing function h.

# In[ ]:


def Solution_2d():
    #用收敛的h函数计算二维分布
    global H
    rho_solution=np.ones(shape=(n_points, n_points))
    for index_x in range(n_points):
        for index_y in range(n_points):
            x=(-V/2+deltaV/2)+index_x*(deltaV)
            y=(-V/2+deltaV/2)+index_y*(deltaV)
            for n in range(Number_constraints):
                    u=A[n][0][0]*x+A[n][0][1]*y                   
                    rho_solution[index_x][index_y]=rho_solution[index_x][index_y]*constant_1d_function(u,H[n])            
    rho_solution=rho_solution/np.sum(rho_solution)  #归一化
    return rho_solution


# ### Difference_2d()：
# ### 使用当前的h函数计算一个二维分布，对这个二维分布在所有约束条件对应的方向投影，计算投影与约束条件的差值的绝对值的平均值
# ### Compute a two-dimensional distribution using the current h function, project this distribution onto all directions corresponding to the constraint conditions, and calculate the average absolute difference between the projection and the constraint conditions.

# In[ ]:


def Difference_2d():
    global Number_constraints
    global H
    global A
    global V
    global deltaV
    global G
    global entropy
    global Transmit_tem
    tem_solution=Solution_2d()
    Transmit_tem={}
    Projection={}
    difference=0
    for i in range(Number_constraints): 
        Transmit_tem[i]=Transmit_2d(A[i], tem_solution)
        Projection[i]=Line(Transmit_tem[i],1)
        matrixA = Projection[i]
        matrixB = G[i]
        matrixA = matrixA / np.sum(matrixA)
        matrixB = matrixB / np.sum(matrixB)
        matrix_difference=np.abs(matrixA-matrixB)
        difference=difference+np.sum(matrix_difference)   
    difference=difference/(2*Number_constraints)
    print("differ=",difference)
    return difference



def Chip_off(function,halo):
    
    a=(halo)*np.max(function)
    function[function < a] = 0
   
    
    return function

#Compare whether matrix A and matrix B are the same after normalization.
def compare(A_name, B_name,relative_error):
    global n_points
    A = eval(A_name)
    B = eval(B_name)
    A = A / np.sum(A)
    B = B / np.sum(B)
    i=6
    print(f"比较{A_name}和{B_name}是否相同")
    
    if np.allclose(A, B, rtol=relative_error):        
        print(f"{A_name}和{B_name}在相对误差允许{100*relative_error}%时全部区域可以认为是相同")
    else:
        A=Chip_off(A,np.e**(-i*1.5))
        B=Chip_off(B,np.e**(-i*1.5))
        if np.allclose(A, B, rtol=relative_error):        
            print(f"{A_name}和{B_name}中大于峰值*exp(-{i*1.5})的中心区域在相对误差允许{100*relative_error}%时可以认为是相同")
        else:
            i=i-1
            A=Chip_off(A,np.e**(-i*1.5))
            B=Chip_off(B,np.e**(-i*1.5))
            if np.allclose(A, B, rtol=relative_error):        
                print(f"{A_name}和{B_name}中大于峰值*exp(-{i*1.5})的中心区域在相对误差允许{100*relative_error}%时可以认为是相同")
            else:
                i=i-1
                A=Chip_off(A,np.e**(-i*1.5))
                B=Chip_off(B,np.e**(-i*1.5))
                if np.allclose(A, B, rtol=relative_error):        
                    print(f"{A_name}和{B_name}中大于峰值*exp(-{i*1.5})的中心区域在相对误差允许{100*relative_error}%时可以认为是相同")
                else:
                    i=i-1
                    A=Chip_off(A,np.e**(-i*1.5))
                    B=Chip_off(B,np.e**(-i*1.5))
                    if np.allclose(A, B, rtol=relative_error):        
                        print(f"{A_name}和{B_name}中大于峰值*exp(-{i*1.5})的中心区域在相对误差允许{100*relative_error}%时可以认为是相同")
                    else:
                        i=i-1
                        A=Chip_off(A,np.e**(-i*1.5))
                        B=Chip_off(B,np.e**(-i*1.5))
                        if np.allclose(A, B, rtol=relative_error):        
                            print(f"{A_name}和{B_name}中大于峰值*exp(-{i*1.5})的中心区域在相对误差允许{100*relative_error}%时可以认为是相同")
                        else:
                            i=i-1
                            A=Chip_off(A,np.e**(-i*1.5))
                            B=Chip_off(B,np.e**(-i*1.5))
                            if np.allclose(A, B, rtol=relative_error):        
                                print(f"{A_name}和{B_name}中大于峰值*exp(-{i*1.5})的中心区域在相对误差允许{100*relative_error}%时可以认为是相同")                           

    error = f"统计{A_name}和{B_name}在所有网格内的差值的积分differ,differ是0到1之间的值\n"
    # 创建一个布尔数组，标记A和B中不同的元素
    A = eval(A_name)
    B = eval(B_name)
    A = A / np.sum(A)
    B = B / np.sum(B)
    #atol=relative_error*np.maximum(np.abs(A),np.abs(B))
    mask = np.abs(A-B)>0
    differ=0

    # 输出不同元素的位置和值
    for index in np.argwhere(mask):
        pos = tuple(index)
        value_A = A[pos]
        value_B = B[pos]
        differ=abs(value_A-value_B)/2+differ
        error += f"位置: {pos}, {A_name}的值: {value_A}, {B_name}的值: {value_B}\n"
        # 将输出保存到文件
    error += f"differ={differ}"
    file_path = f"error{A_name}.txt"
    with open(file_path, "w") as file:
        file.write(error)

    print(f"统计{A_name}和{B_name}所有的比较区别，结果保存在：", file_path)

                                
            
        


# ### 调用Two_to_one_dimensional（）函数获得重构结果; 第一个输入的参数是约束条件G，它是所有约束条件的集合，每个约束条件都是一维的曲线；第二个输入的参数是约束条件的方向A，A[i]对应G[i]的方向，A[i]是2乘2的矩阵，表述从[x,y]方向到G[i]所在的方向的映射，G[i]是二维函数通过映射A[i]后在横轴上的投影；计算出来的结果是归一化的；每轮迭代完成后都重构一个二维图像，在所有A的方向对这个二维图像积分，统计这些积分与约束条件的差值的平均值，记为differ;最终重构的结果保存在"Rho_solution.npy"
# ### To obtain the reconstruction result, you can call the `Two_to_one_dimensional()` function. The function takes two input parameters:
# ### 1. The first parameter is the set of constraint conditions, `G`. Each constraint condition is a one-dimensional curve.
# ### 2. The second parameter is the direction of the constraint conditions, `A`. `A[i]` corresponds to the direction of `G[i]`. `A[i]` is a 2x2 matrix that represents the mapping from the `[x, y]` direction to the direction of `G[i]`.`G[i]` is the projection of the two-dimensional function after mapping `A[i]` onto the horizontal axis.
# ### The computed results are normalized. After each iteration, a two-dimensional image is reconstructed. For this image, we integrate it over all directions of A and calculate the average difference between these integrals and the constraint conditions. This average difference is denoted as "differ". 
# ### Save the reconstruction result in 'Rho_solution.npy'. 

# In[ ]:


def Two_to_one_dimensional(G,A):
    global V
    global n_points
    global deltaV
    global Number_constraints
    global H
    
    #确认不同坐标系之间的关系
    Ajks = np.zeros((Number_constraints,Number_constraints,2,2))   
    for j in range(Number_constraints):
        for k in range(Number_constraints):
            Ajks[j][k]=np.dot(A[j],np.linalg.inv(A[k]))
           

    #为中间过程的h函数声明的储存空间
    H={}
    for i in range(Number_constraints):
        H[i]=np.ones((n_points))
        H[i]=H[i]/np.sum(H[i])
    
    #对h函数做迭代   
    H_iterative=np.ones(shape=(999, Number_constraints, n_points))
    Differ=[]
    Differ.append(Difference_2d())
    for m in range(999):
        for n in range(Number_constraints):
            print(f"进行到第{m+1}轮迭代，正在计算第{n+1}个方向的h函数")           
            for Q in range(n_points):
                u_n=(-V/2+deltaV/2)+Q*(deltaV)
                Multiply=np.ones(n_points)
                #积分
                for N in range(Number_constraints):
                    if n != N:
                        Multiply=np.multiply(rotate(u_n,Ajks[N][n],H[N]),Multiply)                                                                            
                integral=np.sum(Multiply)
                
                if (G[n][Q]==0):
                    H[n][Q]=0
                if (G[n][Q] != 0):
                    if(integral==0):
                        print(f"在计算第{m+1}轮迭代，第{n+1}个方向的h函数时出现奇点！！！！！")                                                                                                                          
                        print(f"奇点的位置：{u_n}")
                        for N in range(Number_constraints):
                            if n != N:
                                plt.plot(rotate(u_n,Ajks[N][n],H[N],n))                            
                                plt.show()                    
                        sys.exit()
                    H[n][Q]=G[n][Q]/integral
                if (H[n][Q]>1e+308):
                     print(H[n][Q],G[n][Q],integral)
                     print("爆炸啦！！！")
                     print("取与约束条件的差值最小的点作为迭代的结果")
                     anchor=np.argmin(Differ)
                     for i in range(Number_constraints):
                         H[i]=H_iterative[anchor-1][i]
                     
                     
                     x = np.arange(0,m+1)
                     plt.plot(x,Differ)
                     plt.xticks(x)
                     plt.xlabel("iterative")
                     plt.ylabel("Differ")
                     plt.show()
                     rho_solution=Solution_2d()
                     print("计算结果已保存在Rho_solution.npy")
                     file_path = "Rho_solution.npy"
                     np.save(file_path,rho_solution)
                     Draw(rho_solution, "x", "y")
                     return
                     sys.exit()
            H[n]=H[n]/np.sum(H[n])
            H_iterative[m][n]=H[n]
        file_path="H_iterative.npy"       
        np.save(file_path,H_iterative)
        Differ.append(Difference_2d())
        if(m>0 and np.allclose(H_iterative[m], H_iterative[m-1], rtol=0.1)):                      
            print(f"进行到第{m+1}轮迭代时收敛了")            
            x = np.arange(0,m+2)
            plt.plot(x,Differ)
            plt.xticks(x)
            plt.xlabel("iterative")
            plt.ylabel("Differ")
            plt.show()
            rho_solution=Solution_2d()
            print("计算结果已保存在Rho_solution.npy")
            file_path = "Rho_solution.npy"
            np.save(file_path,rho_solution)
            Draw(rho_solution, "x", "y")
            return
        if (m>0 and Differ[m]<=Differ[m+1] and Differ[m]<=Differ[m-1]):
            print(f"在第{m}轮迭代获得了与约束条件的差值的极小值点")            
            x = np.arange(0,m+2)
            plt.plot(x,Differ)
            plt.xticks(x)
            plt.xlabel("iterative")
            plt.ylabel("Differ")
            plt.show()
            for i in range(Number_constraints):
                H[i]=H_iterative[m-1][i]
            rho_solution=Solution_2d()
            print("计算结果已保存在Rho_solution.npy")
            file_path = "Rho_solution.npy"
            np.save(file_path,rho_solution)
            Draw(rho_solution, "x", "y")
            return

