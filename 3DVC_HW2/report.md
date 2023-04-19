# 3DVC_HW2

陈新	2022210877	计研三一



## 1. Volume Rendering

Rendering output see in `Problem1/images`

### 1. Ray sampling

![grid_vis](Problem1/images/grid_vis.png)

![ray_vis](Problem1/images/ray_vis.png)



### 2. Point sampling

![point_vis](Problem1/images/point_vis.png)



### 3. Theory of transmittance calculation

$$
\begin{aligned}

T(x_4, x_1) 
&= \prod_{i=1}^3 e^{-\sigma_i \Delta t_i}	\\
&= e^{- \sum_{i=1}^3 \sigma_i \Delta t_i}	\\
&= e^{-31}

\end{aligned}
$$



### 4. Rendering

![render_cube](Problem1/images/render_cube.gif)

<img src="Problem1/images/render_nerf.gif" alt="render_nerf" style="zoom: 200%;" />



## 2. Single Image to 3D

### 1





### 2. Network design






## 3. Surface Reconstruction

### 1. MLS constraints

Set $\epsilon = 0.01$

The outer green points represent constraints (b), and the inner red points represent constraints (c)

![constraints-visualization](Problem3/constraints-visualization.png)



### 2. MLS interpolation





### 3. Marching Cube



