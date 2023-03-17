# 3DVC_HW1

陈新	2022210877	计研三一



## 1. QCQP

1. $\nabla_x L = A^T(A \bullet x - b) +2 \lambda x$

2. $x = (A^T A)^{-1} A^T b$

   $\text{when} \space x^Tx < \epsilon$

3. $x = h(\lambda) = (A^T A + 2 \lambda I)^{-1} A^T b$

   prof:
   $$
   h(\lambda)^T h(\lambda) = b^T A (A^T A + 2 \lambda I)^{-1T} (A^T A + 2 \lambda I)^{-1} A^T b \\
   
   \frac{\partial h(\lambda)^T h(\lambda)}{\partial \lambda} = -4 (A^T b)^T (A^T A + 2 \lambda I)^{-1} A^T b
   $$
   Because $A^T A = U \Lambda U^T$, where $U$ is positive-definite matrix and $\Lambda$ is all-positive diagonal matrix, so
   $$
   \begin{aligned}
   \frac{\partial h(\lambda)^T h(\lambda)}{\partial \lambda}
   &= -4 (A^T b)^T (U \Lambda U^T + 2 \lambda I)^{-1} (A^T b) \\
   &= -4 (A^T b)^T (U(\Lambda  + 2 \lambda U^{-1} U^{T-1})U^T)^{-1} (A^T b) \\
   &= -4 (U^{-1} A^T b)^T (\Lambda  + 2 \lambda (U^{T} U)^{-1})^{-1} (U^{-1} A^T b)
   \end{aligned}
   $$
   Assume $y = U^{-1} A^T b$, then $\frac{\partial h(\lambda)^T h(\lambda)}{\partial \lambda} = -4 y^T (\Lambda  + 2 \lambda (U^{T} U)^{-1})^{-1} y$
   
   Then $\Lambda  + 2 \lambda (U^{T} U)^{-1}$  and $(\Lambda  + 2 \lambda (U^{T} U)^{-1})^{-1}$ is positive-definite matrices when $\lambda \gt 0$.
   
   So $\frac{\partial h(\lambda)^T h(\lambda)}{\partial \lambda} \le 0$ for $\lambda \gt 0$
   
   $h(\lambda)^T h(\lambda)$ is monotonically decreasing for $\lambda \ge 0$

4. Set $\lambda_r - \lambda_l <= 1e-6$ as termination condition, and we get 
   $$
   f(x) = 8.610063646489696
   $$



## 2. 3D Geometry Processing

1. Sample 100K points uniformly on the surface

   <img src="data/saddle_even.jpg" alt="saddle_even" style="zoom: 50%;" />

2. Use iterative farthest point sampling method to sample 4K points from the 100K uniform samples

      <img src="data/saddle_generated.jpg" alt="saddle_generated" style="zoom: 50%;" />

3. Normal estimation

      Use PCA method to fit 50 neighbor points around the target point with `n_components=3`. The first 2 components represent the 2 directions with large variance, namely the fitted plane. The 3rd component with the smallest variance represent the fitted normal. 

      <img src="data/saddle_normal.jpg" alt="saddle_normal" style="zoom:50%;" />

4. 

3. 



## 3. Rotation

1. $|(p+q)/2| = \frac{1}{2} + \frac{1}{8} + \frac{1}{8} = \frac{3}{4}$

   $r = \sqrt{\frac{2}{3}} + \frac{i}{\sqrt{6}} + \frac{j}{\sqrt{6}}$

2. 
