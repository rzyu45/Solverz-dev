from __future__ import annotations

from typing import Union, List, Dict, Callable, Tuple

import numpy as np
import sympy
from sympy import Symbol, preorder_traversal, Basic, Expr, latex, Derivative, sympify, simplify
from sympy import lambdify as splambdify
from sympy.abc import t, x

from Solverz.num.num_alg import F, X, StateVar, AliasVar, AlgebraVar, ComputeParam, new_symbols, \
    pre_lambdify, Mat_Mul, Param_, Var, IdxVar, idx, IdxParam, Const_, IdxConst, minmod, Slice
from Solverz.num.num_interface import numerical_interface
from Solverz.num.matrix_calculus import MixedEquationDiff
from Solverz.param import Param


class Eqn:
    """
    The Equation object
    """

    def __init__(self,
                 name: str,
                 eqn: Expr):

        self.name = name
        self.LHS = 0
        self.RHS = eqn
        self.SYMBOLS: Dict[str, Symbol] = self.obtain_symbols()

        # if the eqn has Mat_Mul, then label it as mixed-matrix-vector equation
        if self.expr.has(Mat_Mul):
            self.mixed_matrix_vector = True
        else:
            self.mixed_matrix_vector = False

        self.NUM_EQN: Callable = self.lambdify()
        self.derivatives: Dict[str, EqnDiff] = dict()

    def obtain_symbols(self) -> Dict[str, Symbol]:
        temp_dict = dict()
        for symbol_ in list((self.LHS - self.RHS).free_symbols):
            if isinstance(symbol_, (Var, Param_, Const_, idx)):
                temp_dict[symbol_.name] = symbol_
            elif isinstance(symbol_, (IdxVar, IdxParam, IdxConst)):
                temp_dict[symbol_.symbol.name] = symbol_.symbol
                if isinstance(symbol_.index, idx):
                    temp_dict[symbol_.index.name] = symbol_.index
                elif isinstance(symbol_.index, tuple):
                    for idx_ in symbol_.index:
                        if isinstance(idx_, idx):
                            temp_dict[idx_.name] = idx_
                elif isinstance(symbol_.index, (slice, Expr)):
                    temp_dict.update(symbol_.symbol_in_index)
                elif isinstance(symbol_.index, list):
                    temp_dict.update(symbol_.symbol_in_index)

        return temp_dict

    def lambdify(self) -> Callable:
        return splambdify(self.SYMBOLS.values(), pre_lambdify(self.RHS), [numerical_interface, 'numpy'])

    def eval(self, *args: Union[np.ndarray]) -> np.ndarray:
        return self.NUM_EQN(*args)

    def derive_derivative(self):
        """"""
        for symbol_ in list(self.RHS.free_symbols):
            # differentiate only to variables
            if isinstance(symbol_, IdxVar):  # if the equation contains Indexed variables
                idx_ = symbol_.index
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                         eqn=diff,
                                                         diff_var=symbol_,
                                                         var_idx=idx_.name if isinstance(idx_, idx) else idx_)
            elif isinstance(symbol_, Var):
                if self.mixed_matrix_vector:
                    diff = MixedEquationDiff(self.RHS, symbol_)
                else:
                    diff = self.RHS.diff(symbol_)
                self.derivatives[symbol_.name] = EqnDiff(name=f'Diff {self.name} w.r.t. {symbol_.name}',
                                                         eqn=diff,
                                                         diff_var=symbol_)

    @property
    def expr(self):
        return self.LHS - self.RHS

    def subs(self, *args, **kwargs):
        return self.RHS.subs(*args, **kwargs)

    def __repr__(self):
        # sympy objects' printing prefers __str__() to __repr__()
        return self.LHS.__str__() + r"=" + self.RHS.__str__()

    def _repr_latex_(self):
        """
        So that jupyter notebook can display latex equation of Eqn object.
        :return:
        """
        return r"$\displaystyle %s$" % (latex(self.LHS) + r"=" + latex(self.RHS))


class EqnDiff(Eqn):
    """
    To store the derivatives of equations W.R.T. variables
    """

    def __init__(self, name: str, eqn: Expr, diff_var: Symbol, var_idx=None):
        super().__init__(name, eqn)
        self.diff_var = diff_var
        self.diff_var_name = diff_var.symbol.name if isinstance(diff_var, IdxVar) else diff_var.name
        self.var_idx = var_idx  # df/dPi[i] then var_idx=i
        self.var_idx_func = None
        if self.var_idx is not None:
            if isinstance(self.var_idx, slice):
                temp = []
                if var_idx.start is not None:
                    temp.append(var_idx.start)
                if var_idx.stop is not None:
                    temp.append(var_idx.stop+1)
                if var_idx.step is not None:
                    temp.append(var_idx.step)
                self.var_idx_func = Eqn('To evaluate var_idx of variable' + self.diff_var.name, Slice(*temp))
            elif isinstance(self.var_idx, Expr):
                self.var_idx_func = Eqn('To evaluate var_idx of variable' + self.diff_var.name, self.var_idx)
        self.LHS = Derivative(sympy.Function('g'), diff_var)


class Ode(Eqn):
    r"""
    The class for ODE reading

    .. math::

         \frac{\mathrm{d}y}{\mathrm{d}t}=f(t,y)

    where $y$ is the state vector.

    """

    def __init__(self, name: str,
                 f: Expr,
                 diff_var: Union[Var, IdxVar]):
        super().__init__(name, f)
        self.diff_var = diff_var
        self.LHS = Derivative(diff_var, t)

    def discretize(self,
                   scheme: Basic,
                   param: Dict[str, Param] = None,
                   extra_diff_var: List[str] = None):
        """

        :param extra_diff_var: diff_var from other Eqn
        :param scheme:
        :param param: list of parameters in the Ode
        :return:
        """

        if not extra_diff_var:
            extra_diff_var = []

        if not param:
            param = dict()

        funcs: Dict[F, Basic] = dict()  # function set
        alias: Dict[AliasVar, Symbol] = dict()
        scheme_elements = preorder_traversal(scheme)
        for arg in scheme_elements:
            if isinstance(arg, F):
                # generate subs dict of FunctionClass F
                # arg is a Function
                f_args = arg.args
                # args of Functions
                symbol_dict: Dict[Symbol, Basic] = dict()
                for symbol in self.SYMBOLS:
                    if symbol.name in [self.diff_var] + extra_diff_var:
                        # State Variable
                        symbol_dict[symbol] = self._subs_state_var_in_func_args(f_args[0], symbol)
                    elif symbol.name == 't':
                        # Time variable of non-autonomous equations
                        if 't' not in param.keys():
                            param['t'] = Param('t')
                        symbol_dict[symbol] = self._subs_t_in_func_args(f_args[2])
                    elif symbol.name not in param.keys():
                        # Algebra Variable
                        symbol_dict[symbol] = self._subs_algebra_var_in_func_args(f_args[1], symbol)
                funcs[arg] = self.subs(symbol_dict)
                scheme_elements.skip()
                # skip the args of function class
            elif isinstance(arg, AliasVar):
                if arg.alias_of == 'X':
                    alias[arg] = new_symbols(self.diff_var + arg.suffix, commutative=self.commutative)
                elif arg.alias_of == 'Y':
                    raise ValueError('Really? Schemes may be wrong.')

        scheme = scheme.subs([(key, value) for key, value in funcs.items()])
        scheme = scheme.subs([(key, value) for key, value in alias.items()])
        scheme = scheme.subs([(X, new_symbols(self.diff_var, commutative=self.commutative))])

        # Add new Param
        for symbol in list(scheme.free_symbols):
            if symbol not in self.SYMBOLS and symbol.name not in param and symbol.name != self.diff_var:
                param[symbol.name] = Param(symbol.name)

        return param, Eqn('d_' + self.name, eqn=scheme.__str__(), commutative=self.commutative)

    def _subs_state_var_in_func_args(self, expr: Basic, symbol: Symbol):
        subs_dict: Dict[Union[StateVar, AliasVar, ComputeParam], Symbol] = dict()
        for symbol_ in list(expr.free_symbols):
            if isinstance(symbol_, StateVar):
                subs_dict[symbol_] = symbol
            elif isinstance(symbol_, AliasVar):
                subs_dict[symbol_] = new_symbols(symbol.name + symbol_.suffix, commutative=self.commutative)
            elif isinstance(symbol_, ComputeParam):
                subs_dict[symbol_] = new_symbols(symbol_.name, commutative=self.commutative)
        return expr.subs(subs_dict)

    def _subs_t_in_func_args(self, expr: Basic):
        # symbol=t
        subs_dict: Dict[Union[ComputeParam], Symbol] = dict()
        for symbol_ in list(expr.free_symbols):
            if isinstance(symbol_, ComputeParam):
                subs_dict[symbol_] = new_symbols(symbol_.name, commutative=self.commutative)
        return expr.subs(subs_dict)

    def _subs_algebra_var_in_func_args(self, expr: Basic, symbol: Symbol):
        subs_dict: Dict[Union[AlgebraVar, AliasVar, ComputeParam], Symbol] = dict()
        for symbol_ in list(expr.free_symbols):
            if isinstance(symbol_, AlgebraVar):
                subs_dict[symbol_] = symbol
            elif isinstance(symbol_, AliasVar):
                subs_dict[symbol_] = new_symbols(symbol.name + symbol_.suffix, commutative=self.commutative)
            elif isinstance(symbol_, ComputeParam):
                subs_dict[symbol_] = new_symbols(symbol_.name, commutative=self.commutative)
        return expr.subs(subs_dict)


class Pde(Eqn):
    """
    The class of partial differential equations
    """
    pass


class HyperbolicPde(Pde):
    r"""
    The class for hyperbolic PDE reading

    .. math::

         \frac{\partial{u}}{\partial{t}}+\frac{\partial{f(u)}}{\partial{x}}=S(u)

    where $u$ is the state vector, $f(u)$ is the flux function and $S(u)$ is the source term.

    Parameters
    ==========

    two_dim_var : Var or list of Var

        Specify the two-dimensional variables in the PDE. Some of the variables, for example, the mass flow $\dot{m}$ in
        the heat transmission equation, are not two-dimensional variables.

    """

    def __init__(self, name: str,
                 diff_var: Var,
                 flux: Expr,
                 source: Expr = 0,
                 two_dim_var: Union[Var, List[Var]] = None):
        if isinstance(source, (float, int)):
            source = sympify(source)
        super().__init__(name, source)
        self.diff_var = diff_var
        self.flux = flux
        self.source = source
        self.two_dim_var = [two_dim_var] if isinstance(two_dim_var, Var) else two_dim_var
        self.LHS = Derivative(diff_var, t) + Derivative(flux, x)

    def derive_derivative(self):
        pass

    def finite_difference(self, scheme=1):
        r"""
        Discretize hyperbolic PDE as AEs.

        Parameters
        ==========

        scheme : int

            1 - Central difference

            .. math::

                \frac{\partial{u}}{\partial{t}}\approx\frac{u_{i+1}^{j+1}-u_{i+1}^{j}+u_{i}^{j+1}-u_{i}^{j}}{2\Delta t}

            .. math::

                \frac{\partial{f(u)}}{\partial{x}}\approx\frac{f(u_{i+1}^{j+1})-f(u_{i}^{j+1})+f(u_{i+1}^{j})-f(u_{i}^{j})}{2\Delta x}

            .. math::

                S(u)\approx S\left(\frac{u_{i+1}^{j+1}+u_{i}^{j+1}+u_{i+1}^{j}+u_{i}^{j}}{4}\right)

            2 - Characteristic line method

        Returns
        =======

        AE : Eqn

            Let's take central difference as an example, this function returns the algebraic equation

            .. math::

                \begin{aligned}
                    0=&\Delta x(\tilde{u}[1:M]-\tilde{u}^0[1:M]+\tilde{u}[0:M-1]-\tilde{u}^0[0:M-1])+\\
                      &\Delta t(f(\tilde{u}[1:M])-f(\tilde{u}[0:M-1])+f(\tilde{u}^0[1:M])-f(\tilde{u}^0[0:M-1]))+\\
                      &2\Delta x\Delta t\cdot S\left(\frac{u_{i+1}^{j+1}+u_{i}^{j+1}+u_{i+1}^{j}+u_{i}^{j}}{4}\right)
                \end{aligned}

            where we denote by vector $\tilde{u}$ the discrete spatial distribution of state $u$, by $\tilde{u}^0$ the
            initial value of $\tilde{u}$, and by $M$ the last index of $\tilde{u}$.

        """
        if scheme == 1:
            dx = Const_('dx')
            dt = Param_('dt')
            M = idx('M')
            u = self.diff_var
            u0 = Param_(u.name + '0')

            fui1j1 = self.flux.subs([(a, a[1:M]) for a in self.two_dim_var])
            fuij1 = self.flux.subs([(a, a[0:M - 1]) for a in self.two_dim_var])
            fui1j = self.flux.subs([(a, Param_(a.name + '0')[1:M]) for a in self.two_dim_var])
            fuij = self.flux.subs([(a, Param_(a.name + '0')[0:M - 1]) for a in self.two_dim_var])

            S = self.source.subs(
                [(a, (a[1:M] + a[0:M - 1] + Param_(a.name + '0')[1:M] + Param_(a.name + '0')[0:M - 1]) / 4) for a in
                 self.two_dim_var])

            ae = dx * (u[1:M] - u0[1:M] + u[0:M - 1] - u0[0:M - 1]) \
                 + simplify(dt * (fui1j1 - fuij1 + fui1j - fuij)) \
                 - simplify(2 * dx * dt * S)

            # ae = (u[1:M] - u0[1:M] + u[0:M - 1] - u0[0:M - 1]) / dt \
            #      + simplify((fui1j1 - fuij1 + fui1j - fuij))/dx\
            #      - simplify(2 * S)

            return Eqn('FDM of ' + self.name, ae)

    def semi_discretize(self, a, scheme=1):
        r"""
        Semi-discretize the hyperbolic PDE of nonlinear conservation law as ODEs using the Kurganov-Tadmor scheme
        (see [Kurganov2000]_). The difference stencil is as follows, with $x_{j+1}-x_{j}=\Delta x$.

            .. image:: ../../pics/difference_stencil.png
               :height: 100

        Parameters
        ==========

        a : List[Expr]

            Maximum local speed $a=[a_{j+1/2}, a_{j-1/2}]$, with formula

            .. math::

                a_{j+1/2}=\max\qty{\rho\qty(\pdv{f}{u}\qty(u^+_{j+1/2})),\rho\qty(\pdv{f}{u}\qty(u^-_{j+1/2}))},

            .. math::

                a_{j-1/2}=\max\qty{\rho\qty(\pdv{f}{u}\qty(u^+_{j-1/2})),\rho\qty(\pdv{f}{u}\qty(u^-_{j-1/2}))},

            where

            .. math::

                \rho(A)=\max_i|\lambda_i(A)|

        scheme : int



        Returns
        =======

        ODE : List[Union[Ode, Eqn]]

            This function returns the for $2\leq j\leq M-2$

            .. math::

                \dv{t}u_j=-\frac{H_{j+1/2}-H_{j-1/2}}{\Delta x}+S(u_j)

            and for $j=1,M-1$

            .. math::

                \dv{t}u_j=-\frac{f(u_{j+1})-f(u_{j-1})}{2\Delta x}+\frac{a_{j+1/2}(u_{j+1}-u_j)-a_{j-1/2}(u_j-u_{j-1})}{2\Delta x}+S(u_j),

            where

            .. math::

                H_{j+1/2}=\frac{f(u^+_{j+1/2})+f(u^-_{j+1/2})}{2}-\frac{a_{j+1/2}}{2}\qty[u^+_{j+1/2}-u^-_{j+1/2}],

            .. math::

                H_{j-1/2}=\frac{f(u^+_{j-1/2})+f(u^-_{j-1/2})}{2}-\frac{a_{j-1/2}}{2}\qty[u^+_{j-1/2}-u^-_{j-1/2}],

            .. math::

                u^+_{j+1/2}=u_{j+1}-\frac{\Delta x}{2}(u_x)_{j+1},\quad u^-_{j+1/2}=u_j+\frac{\Delta x}{2}(u_x)_j,

            .. math::

                u^+_{j-1/2}=u_{j}-\frac{\Delta x}{2}(u_x)_{j},\quad u^-_{j-1/2}=u_{j-1}+\frac{\Delta x}{2}(u_x)_{j-1},

            .. math::

                (u_x)_j=\operatorname{minmod}\qty(\theta\frac{u_j-u_{j-1}}{\Delta x},\frac{u_{j+1}-u_{j-1}}{2\Delta x},\theta\frac{u_{j+1}-u_{j}}{\Delta x}),\quad \theta\in[1,2],

            and by linear extrapolation

            .. math::

                u_0=2u_\text{L}-u_1,\quad u_M=2u_\text{R}-u_{M-1}.


        .. [Kurganov2000] Alexander Kurganov, Eitan Tadmor, New High-Resolution Central Schemes for Nonlinear Conservation Laws and Convection–Diffusion Equations, Journal of Computational Physics, Volume 160, Issue 1, 2000, Pages 241-282, `<https://doi.org/10.1006/jcph.2000.6459>`_

        """

        if not isinstance(a, list):
            raise TypeError("a should be a list of maximum local speed for a_{j+1/2} and a_{j-1/2}!")

        dx = Const_('dx')
        M = idx('M')
        u = self.diff_var

        if scheme == 1:
            # j=1
            # f(u[2])
            fu2 = self.flux.subs([(var, var[2]) for var in self.two_dim_var])
            # f(u[0])=f(2*uL-u[1])
            fu0 = self.flux.subs([(var, var[0]) for var in self.two_dim_var])
            # S(u[1])
            Su1 = self.source.subs([(var, var[1]) for var in self.two_dim_var])
            ode_rhs1 = -simplify((fu2 - fu0) / (2 * dx)) \
                       + simplify((a[0] * (u[2] - u[1]) - a[1] * (u[1] - u[0])) / (2 * dx)) \
                       + simplify(Su1)

            # j=M-1
            # f(u[M])=f(2*uR-u[M-1])
            fum = self.flux.subs([(var, var[M]) for var in self.two_dim_var])
            # f(u[M-2])
            fum2 = self.flux.subs([(var, var[M - 2]) for var in self.two_dim_var])
            # S(u[M-1])
            SuM1 = self.source.subs([(var, var[M-1]) for var in self.two_dim_var])
            ode_rhs3 = -simplify((fum - fum2) / (2 * dx)) \
                       + simplify((a[0] * (u[M] - u[M - 1]) - a[1] * (u[M - 1] - u[M - 2])) / (2 * dx)) \
                       + simplify(SuM1)

            # 2<=j<=M-2
            def ujprime(U: IdxVar, v: int):
                # for given u_j,
                # returns
                # u^+_{j+1/2} case v==0,
                # u^-_{j+1/2} case 1,
                # u^+_{j-1/2} case 2,
                # u^-_{j-1/2} case 3
                if not isinstance(U.index, slice):
                    raise TypeError("Index of IdxVar must be slice object")
                start = U.index.start
                stop = U.index.stop
                step = U.index.step
                var_name = U.symbol.name
                U = Var(var_name)
                Ux = Var(var_name + 'x')

                # u_j
                Uj = U[start:stop:step]
                # (u_x)_j
                Uxj = Ux[start:stop:step]
                # u_{j+1}
                Ujp1 = U[start + 1:stop + 1:step]
                # (u_x)_{j+1}
                Uxjp1 = Ux[start + 1:stop + 1:step]
                # u_{j-1}
                Ujm1 = U[start - 1:stop - 1:step]
                # (u_x)_{j-1}
                Uxjm1 = Ux[start - 1:stop - 1:step]

                if v == 0:
                    return Ujp1 - dx / 2 * Uxjp1
                elif v == 1:
                    return Uj + dx / 2 * Uxj
                elif v == 2:
                    return Uj - dx / 2 * Uxj
                elif v == 3:
                    return Ujm1 + dx / 2 * Uxjm1
                else:
                    raise ValueError("v=0 or 1 or 2 or 3!")

            # j\in [2:M-2]
            Suj = self.source.subs([(var, var[2:M - 2]) for var in self.two_dim_var])
            Hp = (self.flux.subs([(var, ujprime(var[2:M - 2], 0)) for var in self.two_dim_var]) +
                  self.flux.subs([(var, ujprime(var[2:M - 2], 1)) for var in self.two_dim_var])) / 2 \
                 - a[0] / 2 * (ujprime(u[2:M - 2], 0) - ujprime(u[2:M - 2], 1))
            Hm = (self.flux.subs([(var, ujprime(var[2:M - 2], 2)) for var in self.two_dim_var]) +
                  self.flux.subs([(var, ujprime(var[2:M - 2], 3)) for var in self.two_dim_var])) / 2 \
                 - a[1] / 2 * (ujprime(u[2:M - 2], 2) - ujprime(u[2:M - 2], 3))
            ode_rhs2 = -simplify(Hp - Hm) / dx + Suj

            theta = Const_('theta')
            ux = Var(u.name + 'x')
            minmod_rhs = ux[1:M - 1] - minmod(theta * (u[1:M - 1] - u[0:M - 2]) / dx,
                                              (u[2:M] - u[0:M - 2]) / (2 * dx),
                                              theta * (u[2:M] - u[1:M - 1]) / dx)

            return [Ode('SDM of ' + self.name + ' 1', ode_rhs1, u[1]),
                    Ode('SDM of ' + self.name + ' 2', ode_rhs2, u[2:M - 2]),
                    Ode('SDM of ' + self.name + ' 3', ode_rhs3, u[M - 1]),
                    Eqn('SDM of ' + self.name + ' 4', u[M] - 2 * Var(u.name + 'R') + u[M - 1]),
                    Eqn('SDM of ' + self.name + ' 5', u[0] - 2 * Var(u.name + 'L') + u[1]),
                    Eqn('minmod limiter 1 of ' + u.name, minmod_rhs),
                    Eqn('minmod limiter 2 of ' + u.name, ux[0]),
                    Eqn('minmod limiter 3 of ' + u.name, ux[M])]
        elif scheme == 2:
            # 1<=j<=M-1
            # f(u[j+1])
            fu1 = self.flux.subs([(var, var[2:M]) for var in self.two_dim_var])
            # f(u[j-1])
            fu2 = self.flux.subs([(var, var[0:M - 2]) for var in self.two_dim_var])
            # S(u[j])
            Su = self.source.subs([(var, var[1:M - 1]) for var in self.two_dim_var])
            ode_rhs = -simplify((fu1 - fu2) / (2 * dx)) \
                      + simplify((a[0] * (u[2:M] - u[1:M - 1]) - a[1] * (u[1:M - 1] - u[0:M - 2])) / (2 * dx)) \
                      + simplify(Su)

            return [Ode('SDM of ' + self.name + ' 1', ode_rhs, u[1:M - 1]),
                    Eqn('SDM of ' + self.name + ' 2', u[M] - 2 * Var(u.name + 'R') + u[M - 1]),
                    Eqn('SDM of ' + self.name + ' 3', u[0] - 2 * Var(u.name + 'L') + u[1])]
