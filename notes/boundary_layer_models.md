# Equations
Similarity transform:
$$
\xi \ := \ \int_{0}^{x} \rho_{e}(\bar{x}) u_{e}(\bar{x}) \mu_{e}(\bar{x}) \ d\bar{x}, \\ \eta \ := \ \frac{u_{e}}{\sqrt{2 \xi}} \int_{0}^{y}\rho(\bar{y}) \ d\bar{y}.
$$
Introduce new functions $(f, g)$:
$$
  f'(\xi, \eta) \ := \ u(\xi, \eta) / u_{\xi}, \ \ g(\xi, \eta) \ := \ \frac{h(\xi, \eta)}{h_{e}(\xi)},
$$
where the $'$ superscript denotes differentiation with respect to $\eta$. Pressure is constant in the $\eta$ direction:

$$
\frac{\partial p}{\partial \eta} \ = \ 0, \ p(\xi, \eta) \ = \ p_{e}(\xi).
$$

Momentum equation becomes:
$$
\begin{gather}
(Cf'')' \ + \ f f'' \ = \ c_1 \bigg[ \big( f'\big)^{2} \ - \ \frac{\rho_{e}}{\rho}\bigg] \ + \ 2 \xi \bigg[  f' \frac{\partial f'}{\partial \xi}\ - \ f'' \frac{\partial f}{\partial \xi}\bigg], \\ 
c_{1}(\xi) \ := \ \frac{2 \xi}{u_{e}}\frac{du_{e}}{d\xi}, \ \ C \ := \ \frac{\rho \mu}{\rho_{e} \mu_{e}}.
\end{gather}
$$
where C denotes the Chapman-Rubesin factor. 
Energy equation becomes
$$
\begin{gather}
\bigg( \frac{C}{Pr} g' \bigg)' \ + \ f g' \ = \ f' \bigg( c_{2} g \ + \ c_{3} \frac{\rho_{e}}{\rho} \bigg) \ + \  2 \xi \bigg[ f' \frac{\partial g}{\partial \xi}  \ - \ g' \frac{\partial f}{\partial \xi} \bigg] \ - \ C \frac{u_{e}^2}{h_{e}} \big( f''\big)^{2}, \\
c_{2}(\xi) \ := \ \frac{2\xi}{h_{e}}\frac{dh_{e}}{d\xi}, \ \ c_{3}(\xi) \ := \ u_{e}\frac{2 \xi}{h_{e}}\frac{d u_{e}}{d\xi}.
\end{gather}
$$

Note the relation $c_{2} = - c_{3}$ since total enthalpy is conserved along the edge of the boundary layer. The energy equation simplifies to:

$$
\begin{gather}
\bigg( \frac{C}{Pr} g' \bigg)' \ + \ f g' \ = \ f' c_{2} \bigg( g \ - \ \frac{\rho_{e}}{\rho} \bigg) \ + \  2 \xi \bigg[ f' \frac{\partial g}{\partial \xi}  \ - \ g' \frac{\partial f}{\partial \xi} \bigg] \ - \ C \frac{u_{e}^2}{h_{e}} \big( f''\big)^{2}, 
\end{gather}
$$


Difference-Differential references:
- Hartree, D. R., and Womersley, J. R., “A Method for the Numerical or Mechanical
Solution of Certain Types of Partial Differential Equations,” Proceedings of the
Royal Society, Vol. 161A, Aug. 1937, p. 353.
- Smith, A. M. O., and Clutter, D. W., “Machine Calculation of Compressible Boundary
Layers,” AIAA Journal, Vol. 3, No. 4, April 1965, pp. 639–647.

## Self-Similar Version

Take away the dependency of both $(f, g)$ and the edge conditions on $\xi$. The equations simplify to:

$$
(Cf'')' \ + \ f f'' \ = \ 0, \\
\bigg( \frac{C}{Pr} g' \bigg)' \ + \ f g' \ = \  - \ C \frac{u_{e}^2}{h_{e}} \big( f''\big)^{2}.
$$

## Local-Similarity Equations

Take away the gradients of $(f,g)$ with respect to $\xi$. The equations simplify to:

$$
(Cf'')' \ + \ f f'' \ = \ c_{1} \bigg(  \big(f'\big)^{2} \ - \ \frac{\rho_{e}}{\rho} \bigg), \\
\bigg( \frac{C}{Pr} g' \bigg)' \ + \ f g' \ = \ f' c_{2} \bigg( g \ - \ \frac{\rho_{e}}{\rho} \bigg) \ - \ C \frac{u_{e}^2}{h_{e}} \big( f''\big)^{2},
$$

References:
-  Kemp, N. H., Rose, R. H., and Detra, R. W., “Laminar Heat Transfer Around Blunt
Bodies in Dissociated Air,” Journal of the Aerospace Sciences, Vol. 26, No. 7, July
1959, pp. 421–430.

## Difference-Differential Equations

The model is obtained after applying difference formulas to $\xi-$ derivative terms. We are still solving ODEs in the $\eta$ direction, but now the source terms depend the previously solved profiles. In the momentum equation, we have the term:

$$

2 \xi \bigg[  f' \frac{\partial f'}{\partial \xi}\ - \ f'' \frac{\partial f}{\partial \xi}\bigg]

$$

In the energy equation, we have the term:

$$

2 \xi \bigg[ f' \frac{\partial g}{\partial \xi}  \ - \ g' \frac{\partial f}{\partial \xi} \bigg]
$$


Let the superscript ${}^{(n)}$ refer to the $\xi$-station at which a profile $(Cf'', f', f, (C/Pr)g', g)^{(n)}$ is solved. For simplicity, let's assume a constant $\Delta \xi$. The first-order and second-order backward approximations yield for $f$ (as well as $f'$ and $g$) :

$$
\bigg(\frac{\partial f}{\partial \xi}\bigg)_{BE}^{(n)} \ = \ \frac{{f}^{(n)} -{f}^{(n-1)}}{\Delta \xi}, \ \ \bigg(\frac{\partial f}{\partial \xi}\bigg)_{BDF}^{(n)} \ = \ \frac{3{f}^{(n)} - 4{f}^{(n-1)} + 2{f}^{(n-2)}}{2\Delta \xi}.
$$

A more general second-order formula using Lagrangian interpolation:

$$

\bigg(\frac{\partial f}{\partial \xi}\bigg)_{BDF}^{(n)} \ = \ \frac{2\xi^{n} - \xi^{n-1} - \xi^{n-2}}{(\xi^{n} - \xi^{n-1})(\xi^{n} - \xi^{n-2})} f^{(n)} \ - \ \frac{\xi^{n}  - \xi^{n-2}}{(\xi^{n} - \xi^{n-1})(\xi^{n-1} - \xi^{n-2})} f^{(n-1)} \ \\ + \ \frac{\xi^{n}  - \xi^{n-1}}{(\xi^{n} - \xi^{n-2})(\xi^{n-1} - \xi^{n-2})} f^{(n-2)}.

$$

In compact form, we can write:

$$
2 \xi \frac{\partial f'}{\partial \xi} \ \approx \ m_0 f' \ + \ m_{1}, \ \ 2 \xi \frac{\partial f}{\partial \xi} \ \approx \ s_0 f \ + \ s_{1}, \ \ 2 \xi \frac{\partial g}{\partial \xi} \ \approx \ e_0 g \ + \ e_{1}, 
$$

so that we approximate the momentum source term as:

$$
2 \xi \bigg[  f' \frac{\partial f'}{\partial \xi}\ - \ f'' \frac{\partial f}{\partial \xi}\bigg] \ \approx \ f' \big( m_{0} f' \ + \ m_{1} \big) \ - \ f'' \big( s_{0} f  \ + \ s_{1} \big),
$$

and the energy source term as:

$$

2 \xi \bigg[ f' \frac{\partial g}{\partial \xi}  \ - \ g' \frac{\partial f}{\partial \xi} \bigg] \ \approx \ f' \big( e_{0} g \ + \ e_{1}) \ - \ g' \big( s_{0} f \ + \ s_{1} \big), 
$$

where $\big(m_{ij}\big)_{i,j \in [0,1]^2}$ and $\big( e_{ij} \big)_{i,j \in [0,1]^2}$ are momentum and energy coefficients determined by the approximation formula used and profile data at previous stations $(n-1), (n-2)$. Ultimately, the model writes:

$$
(Cf'')' \ + \ f f'' \ = \ c_{1} \bigg(  \big(f'\big)^{2} \ - \ \frac{\rho_{e}}{\rho} \bigg) + f' \big( m_{0} f' \ + \ m_{1} \big) \ - \ f'' \big( s_{0} f  \ + \ s_{1} \big) , \\
\bigg( \frac{C}{Pr} g' \bigg)' \ + \ f g' \ = \ f' c_{2} \bigg( g \ - \ \frac{\rho_{e}}{\rho} \bigg) \ - \ C \frac{u_{e}^2}{h_{e}} \big( f''\big)^{2} \ + \ f' \big( e_{0} g \ + \ e_{1}) \ - \ g' \big( s_{0} f \ + \ s_{1} \big).
$$
# Case Setup

In terms of grid, you likely start with grid points along the $x$ (body surface) coordinate, and flow conditions $(u_{e}, p_{e}, h_{e})$.

Map $x$-grid to $\xi$-grid, compute the gradients:
$$
\frac{\partial \xi}{\partial x} \ = \ \rho_{e} u_{e} \mu_{e} \ \implies \ \frac{du_{e}}{d\xi} \ = \ \bigg( \frac{d\xi}{dx}\bigg)^{-1} \frac{du_{e}}{dx}, \ \frac{dh_{e}}{d\xi} \ = \ \bigg( \frac{d\xi}{dx}\bigg)^{-1} \frac{dh_{e}}{dx} 
$$

# Calculation of Output Fields

After running the algorithm, we have $\big(C f'',\  f', \ f, \ (C/Pr) g', \ g \big)$ as various $(\xi, \eta)$ locations. We readily get the velocity and enthalpy profiles from $f'$ and $g$. Other quantities of interest are the heat and wall fluxes as well as the $y$ coordinates (boundary layer thickness).

## Heat and viscous fluxes
Let's work out the formulas for the viscous and heat fluxes. Define
$$
F''(\xi, \eta) \ := \ C f''(\xi, \eta), \\  G'(\xi, \eta) \ := \ (C/Pr) g'(\xi, \eta).
$$

The viscous flux $\tau$ is given by:

$$
\begin{equation}
\tau \ := \ \mu \frac{\partial u}{\partial y} \ = \ \mu u_{e} \frac{\partial f'}{\partial y} \ = \ \rho \mu \frac{u_{e}^{2}}{\sqrt{2 \xi}} \frac{\partial f'}{\partial \eta} \ = \ \frac{ \rho_e \mu_e u_e^{2}}{\sqrt{2 \xi}} F''(\xi, \eta) \ = \ u_{e} \bigg[\bigg(\frac{\partial \xi}{\partial x}\bigg) / \sqrt{2 \xi} \bigg]  F''(\xi, \eta). 
\end{equation}
$$

The heat flux $q$ is given by:

$$
\begin{equation}
  q \ := \ k \frac{\partial T}{\partial y} \ = \ \frac{k}{c_p} h_{e} \frac{\partial g}{\partial y} \ = \ \frac{\rho u_{e}}{\sqrt{2 \xi}} \frac{k}{c_{p}} h_{e} \frac{Pr}{C} G'(\xi, \eta) \ = \ h_{e} \bigg[\bigg(\frac{\partial \xi}{\partial x}\bigg) / \sqrt{2 \xi} \bigg]  G'(\xi, \eta). 
\end{equation}
$$

## Profile thickness
The $y$ coordinate is implicity defined by $\eta$

$$
\begin{equation}
\eta \ = \ \frac{ u_{e}}{\sqrt{2 \xi}} \int_{0}^{y} \rho(\bar{y}) \ d\bar{y}
\end{equation}
$$

Given an $\eta$ - grid, we can approximate the corresponding $y$ - grid using a mid-point rule:

$$
\begin{gather}
\big(\Delta y \big)_{n \ + \ \frac{1}{2}}  \ = \ \bigg( \frac{\sqrt{2 \xi}}{ u_{e}} \bigg) \bigg(  \frac{2}{\rho^{n} \ + \ \rho^{n+1}}\bigg) \ \big(\Delta \eta\big)_{n + \frac{1}{2}}
\end{gather}
$$

## Singularities
Formulas (1) through (3) are valid for $\xi > 0$. At $\xi = 0$, singularities arise that can be managed on a case-by-case basis.

For stagnation flow, we can assume $u_{e} = Kx$, $h_{e} = cst$ and $\rho_{e} = cst$ near $\xi = 0$. This gives:

$$
\xi(x) \ = \ \frac{1}{2} \rho_{e} \mu_{e} K x^2 \ \implies \ \frac{\partial \xi}{\partial x} \ = \ \rho \mu_{e} K x
$$

The multiplying factors in (1) to (3) simplify to:

$$
\begin{align}
  u_{e} \bigg[\bigg(\frac{\partial \xi}{\partial x}\bigg) / \sqrt{2 \xi} \bigg] \ =& \ \ \big( \rho_{e} \mu_{e}\big)^{1/2} K^{3/2} x, \\
  h_{e} \bigg[\bigg(\frac{\partial \xi}{\partial x}\bigg) / \sqrt{2 \xi} \bigg] \ =& \ \ h_{e} ( \rho_{e} \mu_{e}\big)^{1/2} K^{1/2}, \\
  \frac{u_{e}}{\sqrt{2 \xi}} \ =& \ \ \bigg( \frac{ K}{ \rho_{e} \mu_{e}} \bigg)^{1/2}.
\end{align}
$$

