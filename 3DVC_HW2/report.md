# 3DVC_HW2

陈新	2022210877	计研三一



## 1. Volume Rendering

Rendering output files in `Problem1/images`

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

### 1. Distance

Positive: $h(X, Y) \ge 0$

Prof:
$$
\begin{aligned}

\because \space & d(x_i, q_j) = || x_i - x_j ||_2 >= 0	\\
\therefore \space & d(X, Y) = \max_{x_i \in X} { [\min_{y_j \in Y} { [d(x_i, y_j)] }] } \ge 0	\\
& \text{Similarly} \space d(Y, X) \ge 0	\\
\therefore \space & h(X, Y) = \max { \{ d(X, Y), d(Y, X) \} } \ge 0

\end{aligned}
$$


Reflexive: $h(X, Y) = 0 \rightarrow X = Y$

Prof:
$$
\begin{aligned}

\because \space & h(X, Y) = \max { \{ d(X, Y), d(Y, X) \} } = 0	\\
\therefore \space & d(X, Y) = d(Y, X) = 0	\\
\because \space & d(X, Y) = \max_{x_i \in X} { [\min_{y_j \in Y} { [d(x_i, y_j)] }] } = 0	\\
\therefore \space & \forall x_i \in X, \min_{y_j \in Y} { [d(x_i, y_j)] } = 0	\\
\therefore \space & \forall x_i \in X, \exists y_j \in Y, s.t. x_i = x_j	\\
& \text{Similarly} \space \forall y_i \in Y, \exists x_j \in X, s.t. x_i = y_j	\\
\therefore \space & X = Y

\end{aligned}
$$


Triangular inequation: $h(A, C) \le h(A, B) + h(B, C)$

Prof:
$$
\begin{aligned}

h(A, B) + h(B, C)
&= \max { \{ d(A, B), d(B, A) \} } + \max { \{ d(B, C), d(C, B) \} }	\\
& \ge d(B, A) + d(B, C)	\\
&= \max_{b_i \in B} { [\min_{a_j \in A} { [d(b_i, a_j)] }] } + \max_{b_i \in B} { [\min_{c_j \in C} { [d(b_i, c_j)] }] }	\\
&= \max_{b_i \in B} { \{ \min_{a_j \in A} { [d(b_i, a_j)] } + \min_{c_j \in C} { [d(b_i, c_j)] } \} }	\\
& \ge \max_{b_i \in B} { [ \min_{a_j \in A, c_j \in C} { [ d(b_i, a_j) + d(b_i, c_j)] }] }	\\
& \ge \min_{a_j \in A, c_j \in C} { [ d(a_i, c_j)] }	\\
& \ge h(A, C)

\end{aligned}
$$



### 2. Network design






## 3. Surface Reconstruction

### 1. MLS constraints

Set $\epsilon = 0.01$

The outer green points represent constraints (b), and the inner red points represent constraints (c)

![constraints-visualization](Problem3/constraints-visualization.png)



### 2. MLS interpolation

For each dimension: 

​	Voxel number =  $(38, 37, 30)$

​	Voxel size = $(0.00461889, 0.00466862, 0.0046734)$



I used 3 weight functions:
$$
\begin{aligned}

\theta_{Gaussian}(r) &= e^{- \frac{r^2}{2h^2}}	\\
\theta_{Wendland}(r) &= (1 - \frac{r}{h})^4 (\frac{4r}{h} + 1)	\\
\theta_{Singular}(r) &= \frac{1}{r^2 + h^2}

\end{aligned}
$$
$h \in (0.1, 0.01, 0.001)$

n_neighbors $\in (50, 200, 500, 1000)$

Outputs visualized in the next section. 



### 3. Marching Cube

As is stated in the last section, outputs of the combination (added $k \in (0, 1, 2)$) are listed in directories `./outputs-guassian_fn`, `./outputs-wendland_fn`, `./outputs-singular_fn`

The selection of $\theta$ function seems to have limited impact on the output, so we take $\theta_{Gaussian}$ as an example. 



Here are selected results: 

| k    | h    | n_neighbors | result                                                       |
| ---- | ---- | :---------- | ------------------------------------------------------------ |
| 0    | 0.1  | 200         | ![bunny-mls-k=0-h=0.1-n_neighbors=200](Problem3/pictures/bunny-mls-k=0-h=0.1-n_neighbors=200.png) |
| 0    | 0.01 | 1000        | ![bunny-mls-k=0-h=0.01-n_neighbors=1000](Problem3/pictures/bunny-mls-k=0-h=0.01-n_neighbors=1000.png) |
| 1    | 0.1  | 200         | ![bunny-mls-k=1-h=0.1-n_neighbors=200](Problem3/pictures/bunny-mls-k=1-h=0.1-n_neighbors=200.png) |
| 2    | 0.1  | 200         | ![bunny-mls-k=2-h=0.1-n_neighbors=200](Problem3/pictures/bunny-mls-k=2-h=0.1-n_neighbors=200.png) |



Overall, results of $h = 0.001$ are unacceptable. The best n_neighbors is 200 or 500, otherwise the surface would be too smooth or too sharp. 



Although $k = 1$ is much smoother, it will eliminate high-frequency details, like dents in the eyes and neck. And the problem of surface sharpness could be solved by normal interpolation in rendering pipelines. So $k = 0$ is generally better than $k = 1$. 



$k=2$ is much worse, because it brings floating voxels at the corners/edges. I suppose it is because the MLS algorithm use LS to calculate argmin, which doesn't guarantee $f(c_i) = f_i$. When the polynomial order $k$ increases, the edge value of the function will become uncontrollable.  
