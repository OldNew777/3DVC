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
   
   \frac{\partial h(\lambda)^T h(\lambda)}{\partial \lambda} = -4 (A^T b)^T (A^T A + 2 \lambda I)^{-1} A^T b \\
   
   \text{when} \space \lambda \ge 0, A^T A = U \Lambda U^T \space \text{where} \space U = \\
   
   \frac{\partial h(\lambda)^T h(\lambda)}{\partial \lambda} = -4  < 0
   $$
   $A^T A$ is a positive semidefinite matrix, then $A^T A + 2 \lambda I$  and $(A^T A + 2 \lambda I)^{-1}$ is positive semidefinite matrices when $\lambda \gt 0$.
   
   So $\frac{\partial h(\lambda)^T h(\lambda)}{\partial \lambda} \le 0$ for $\lambda \ge 0$
   
   $h(\lambda)^T h(\lambda)$ is monotonically decreasing for $\lambda \ge 0$



## 2. 3D Geometry Processing



## 3. Rotation

1. $|(p+q)/2| = \frac{1}{2} + \frac{1}{8} + \frac{1}{8} = \frac{3}{4}$

   $r = \sqrt{\frac{2}{3}} + \frac{i}{\sqrt{6}} + \frac{j}{\sqrt{6}}$
