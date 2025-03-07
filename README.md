# system-identification
System identification of a system

This repo follows the theories:

Perform $m$ different experiments simultaneously. Advantages compare to excite one after the other:
(i) for the same frequency resolution and input rms value, the signal-to-noise ratio (SNR) is $\sqrt{m}$ times larger, or for the same SNR and input rms value, the measurement time is $m$ times shorter.
(ii) the experiments mimic the operational conditions, which might be a problem if the system behaves nonlinearly.

In the noiseless case, the relation between the input and output DFT spectra of the $m$ experiments is
$$
Y_0\left(k\right) = G_0 \left(j \omega_k\right) U_0 \left(k\right)
$$
with $U_0\left(k\right) \in \mathbb{C}^{n_u \times m}$, $Y_0\left(k\right) \in \mathbb{C}^{n_y \times m}$. It is clear that solving the above equation for $G_0\left(j \omega_k\right)$ puts a strong condition on the excitation design: the matrix $U_0\left(k\right)$ should be regular for all $k$.

In the noisy case the frequency response matrix (FRM) estimate is then obtained as
$$
\hat{G}\left(j \omega_k\right) = Y\left(k\right) U\left(k\right)^{\dagger}.
$$

The covariance of the FRM estimate is related to the input-output noise covariance as
$$
\mathrm{Cov}\left(\mathrm{vec} \left(\hat{G}\left(j \omega_k\right)\right) \right) \approx \left(\overline{U_0\left(k\right) U_0\left(k\right)^{\mathrm{H}}}\right)^{-1} \bigotimes \left(V\left(k\right) C\left(k\right)  V\left(k\right)^{\mathrm{H}} \right),
$$
where $\left(\cdot\right)^{\mathrm{H}}$ denotes the Hermitian transpose,
$$
\left(\cdot\right)^{\mathrm{H}} = \left(\bar{\cdot}\right)^{\mathrm{T}},~\left(\cdot\right)^{-\mathrm{H}} = \left(\cdot^{-1}\right)^{\mathrm{H}}.
$$ And
$$
V\left(k\right) = 
\begin{bmatrix}
I_{n_y} -G_0\left(j \omega_k\right)
\end{bmatrix}
$$
and
$$
C\left(k\right) = 
\begin{bmatrix}
C_Y\left(k\right) &C_{YU}\left(k\right)\\
C_{YU}\left(k\right)^{\mathrm{H}} &C_U\left(k\right)
\end{bmatrix}
$$
with $\bigotimes$ the Kronecker product and $C_Y\left(k\right)$, $C_U\left(k\right)$ and $C_{YU}\left(k\right)$ the input-output noise covariance matrices of one experiment (one column of $Y\left(k\right)$ and $U\left(k\right)$)

Assume that we have $m$ different experiments and each experiment repeats $p$ times.
$$
\begin{align*}
C_{Y\left[i, j\right]}\left(k\right) &= \mathbb{E}\left[\left(Y^i\left(k\right) - \mathbb{E} \left( Y^i\left(k\right) \right)\right)  \overline{\left(Y^j\left(k\right) - \mathbb{E} \left( Y^j\left(k\right) \right)\right)} \right] \\
&= \frac{1}{p-1} \sum_{l=1}^p \left(Y^{i, l}\left(k\right) -  \overline{Y}^i\left(k\right) \right) \overline{\left(Y^{j, l}\left(k\right) -  \overline{Y}^j \left(k\right) \right)}
\end{align*}
$$

$$
C_{U\left[i, j\right]}\left(k\right) = \frac{1}{p-1} \sum_{l=1}^p \left(U^{i, l}\left(k\right) -  \overline{U}^i\left(k\right) \right) \overline{\left(U^{j, l}\left(k\right) -  \overline{U}^j \left(k\right) \right)}
$$

$$
C_{YU\left[i, j\right]}\left(k\right) = \frac{1}{p-1} \sum_{l=1}^p \left(Y^{i, l}\left(k\right) -  \overline{Y}^i\left(k\right) \right) \overline{\left(U^{j, l}\left(k\right) -  \overline{U}^j \left(k\right) \right)}
$$
where 
$$
\overline{Y}^i\left(k\right)  = \frac{1}{p} \sum_{l=1}^p Y^{i, l} \left(k\right),~\overline{U}^i\left(k\right)  = \frac{1}{p} \sum_{l=1}^p U^{i, l} \left(k\right),
$$