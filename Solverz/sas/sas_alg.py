from __future__ import annotations

from functools import reduce
from typing import Union, Type, Dict, Callable, List

from sympy import Symbol, Expr, Add, Mul, Number, sin, Derivative, Pow, cos, Function, Integer, preorder_traversal, \
    Basic, sympify, Float, Matrix, ImmutableDenseMatrix, simplify, latex, lambdify

dfunc_mapping: Dict[Type[Expr], Callable] = {}


class Index(Symbol):
    r"""
    The Index of DT.

    For example, the $k$-th order DT of $x(t)$ is $x[k]$.

    """
    is_Integer = True

    def __new__(cls, name: str, commutative=False):
        obj = Symbol.__new__(cls, f'{name}')
        obj.is_Integer = True
        obj.name = f'{name}'
        return obj


class Slice(Symbol):
    r"""
    The Slice of DT, which is a vector.

    Examples
    ========

    Suppose the start is $k_0$ and the end is $k_1$, then $x[k_0:k_1]$ denotes the vector

    .. math::

        \begin{bmatrix}
        x[k_0]&\cdots &x[k_1]
        \end{bmatrix}

    Parameters
    ==========

    start : start of DT slice
    end   : start of DT slice

    """

    def __new__(cls, start: Union[int, Index, Expr], end: Union[int, Index, Expr], commutative=False):
        if isinstance(start, (Index, Expr)):
            if not all([isinstance(arg, (Number, Index)) for arg in start.free_symbols]):
                raise TypeError('Unsupported slice start!')
        else:
            if start < 0:
                raise ValueError('Slice start should > 0!')
        if isinstance(end, (Index, Expr)):
            if not all([isinstance(arg, (Number, Index)) for arg in end.free_symbols]):
                raise TypeError('Unsupported slice end!')
        else:
            if end < 0:
                raise ValueError('Slice end should > 0!')
        obj = Symbol.__new__(cls, f'{start}:{end}', commutative=commutative)
        obj.start = start
        obj.end = end
        obj.name = f'{start}:{end}'
        return obj


class DT(Symbol):
    """
    The DT object
    """
    __slots__ = ('index', 'name', 'symbol', 'symbol_name')

    def __new__(cls, symbol, index: Union[int, Index, Slice, Type[Expr]], commutative=True):
        if isinstance(index, int) and index < 0:
            raise IndexError("Invalid DT order")
        obj = Symbol.__new__(cls, f'{symbol.name}[{index}]', commutative=commutative)
        obj.symbol = symbol
        obj.index = index
        obj.symbol_name = symbol.name
        obj.name = f'{symbol.name}[{index}]'
        return obj

    def _latex(self, printer):
        return printer._print(self.symbol) + r'\left [' + f'{self.index}' + r'\right ]'


class phi(Symbol):
    """
    The symbol used to denote `sin` function in expressions.
    """

    def __new__(cls, node, commutative=True):
        obj = Symbol.__new__(cls, 'phi', commutative=commutative)
        obj.eqn = node
        obj.name = 'phi' + r'_' + obj.eqn.__repr__()
        return obj

    def _hashable_content(self):
        return self.name, self.eqn

    def _latex(self, printer):
        return r'\phi_\text{%s}' % self.eqn.__repr__()


class psi(Symbol):
    """
    The symbol used to denote `cos` function in expressions.
    """

    def __new__(cls, node, commutative=True):
        """
        Parameters
        ==========

        node : Expr or Symbol

            arg of ``cos`` function.

        """
        obj = Symbol.__new__(cls, 'psi', commutative=commutative)
        obj.eqn = node
        obj.name = 'psi' + r'_' + obj.eqn.__repr__()
        return obj

    def _hashable_content(self):
        return self.name, self.eqn

    def _latex(self, printer):
        return r'\psi_\text{%s}' % self.eqn.__repr__()


class Constant(Symbol):
    """
    The symbol used to denote constants in expressions.
    """

    def __new__(cls, symbol, commutative=False):
        obj = Symbol.__new__(cls, symbol.name, commutative=symbol.is_commutative)
        obj.symbol = symbol
        obj.name = symbol.name
        return obj


def implements_dt_algebra(sym_expr: Type[Expr]):
    """Register an DT function implementation for sympy Expr."""

    def decorator(func):
        dfunc_mapping[sym_expr] = func
        return func

    return decorator


@implements_dt_algebra(Add)
class dAdd(Function):
    r"""
    This function returns DT of addition.

    For expression $\sum_{i}x_i(t)$,

    if the order of DT is an `Index` object `$k$`, it returns

    $$\sum_{i}x_i[k]$$

    Else if the order of DT is a `Slice` object `$k_0:k_1$`, it returns

    .. math::

        \sum_{i}x_i[k_0:k_1]
    """

    @classmethod
    def eval(cls, k, *args):
        if isinstance(k, (Index, Slice, Expr, Integer)):
            if isinstance(k, Expr) and not isinstance(k, (Slice, Integer)):
                # in sympy, Integer objects are instances of Expr
                if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                    raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            return Add(*[_dtify(arg, k) for arg in tuple(args)])
        else:
            raise ValueError('Non DT index input!')


@implements_dt_algebra(Mul)
class dMul(Function):
    r"""
    This function returns the DT of multiplications.

    For expression $\prod_{i=1}^nx_i$,

    if the order of DT is an `Index` object `$k$`, it returns

    $$\bigotimes_{i=1}^n x_i[0:k]$$

    Else if the order of DT is a `Slice` object `$k_0:k_1$`, it returns

    .. math::

        \begin{cases}
          \overline{\bigotimes}_{i=1}^nx_i[k_0:k_1] & k_0=0 \\
          x_1[0:k_1-k_0]\overline{\bigotimes} y[k_0:k_1]+\sum_{i=0}^{k_0-1}x_1[k_0-i:k_1-i]*y[i] & k_0\geq 1
        \end{cases}

    where  $y(t)=\prod_{i=2}^nx_i(t)$.

    .. note::

        The convolution of numbers/constants and variables equals the dot product. For example, if ``a`` is a
        constant and ``x`` is a variable, we want ``dtify(a*x)`` to produce $a*x[0:k]$ instead of
        $a[0:k]\otimes x[0:k]$ because $a[k]=0\ (k\geq 1)$.

        This is accomplished by declare :py:class:`~.Constant` symbol ``a``. When performing :py:class:`~.dMul`,
        we always extract Number and Constant from the Expr $\prod_{i=1}^nx_i(t)$ first. For example, if $x_3$ is a constant
        then ``dMul`` returns ``dMul(x1*x2,k)*x3*dMul(x4*x5*...*xn,k)``

    """

    @classmethod
    def eval(cls, k, *args):
        i = -1
        for arg in args:
            i = i + 1
            if isinstance(arg, (Number, Constant)):
                if i == 0:
                    return Mul(args[0], _dtify(Mul(*args[1:]), k))
                elif i == len(args) - 1:
                    return Mul(_dtify(Mul(*args[:-1]), k), args[-1])
                else:
                    return Mul(_dtify(Mul(*args[:i]), k), args[i], _dtify(Mul(*args[i + 1:]), k))

        if isinstance(k, (Expr, Index)) and not isinstance(k, (Integer, Slice)):
            if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            return dConv_s(*[_dtify(arg, Slice(0, k)) for arg in tuple(args)])
        elif isinstance(k, Slice):
            # Note that k = k.start:k.end here
            if k.start >= 1:
                temp = dConv_v(_dtify(args[0], Slice(0, k.end - k.start)), _dtify(Mul(*args[1:]), k))
                for i in range(k.start):
                    temp = temp + _dtify(args[0], Slice(k.start - i, k.end - i)) * _dtify(Mul(*args[1:]), i)
                return temp
            else:
                return dConv_v(_dtify(args[0], k), _dtify(Mul(*args[1:]), k))
        else:  # integer
            if k >= 1:
                return dConv_s(*[_dtify(arg, Slice(0, k)) for arg in tuple(args)])
            else:  # k==0
                return Mul(*[_dtify(arg, 0) for arg in tuple(args)])


@implements_dt_algebra(sin)
class dSin(Function):
    r"""
    This function returns DT of $\phi(t)=\sin(x(t))$, which is denoted by $\phi[k]$.

    For expression $\phi(t)=\sin(x(t))$,

    if the order of DT is an `Index` object $k$, it returns

    .. math::
        \begin{cases}
            \sum_{m=0}^{k-1} \frac{k-m}{k}\psi[m]x[k-m]=\left(\frac{k-(0:k-1)}{k}*\psi[0:k-1]\right)\otimes x[1:k]& k\geq 2\\
            \psi[0]* x[1]& k= 1\\
            \phi[0]& k=0
        \end{cases}

    Else if the order of DT is a `Slice` object $k_0:k_1$, it returns

    .. math::

        \begin{cases}
          \left(\frac{k_1-(0:k_1-k_0)}{k_1}*\psi[0:k_1-k_0]\right)\overline{\otimes}
          x[k_0:k_1]+\sum_{i=0}^{k_0-1}\left(\frac{k_1-(k_0-i:k_1-i)}{k_1}*\psi[k_0-i:k_1-i]\right)* x[i] & k_0\geq 1 \\
          \left(\left(\frac{k_1-(0:k_1)}{k_1}*\psi[0:k_1]\right)\overline{\otimes} x[0:k_1]\right)*(1-\delta[0:k_1])+\phi[0]*\delta[0:k_1] & k_0=0
        \end{cases}

    Explanation
    ===========

    .. math::

        \begin{aligned}
        \phi[k]=&\left(\frac{k-(0:k-1)}{k}*\psi[0:k-1]\right)\otimes x[1:k]\\
               =&\left(\frac{k-(0:k)}{k}*\psi[0:k]\right)\otimes x[0:k]\quad (k\geq 1).
        \end{aligned}

    See Also
    ========

    dCos

    """

    @classmethod
    def eval(cls, k, *args):
        if len(args) > 1:
            raise ValueError(f'Sin supports one operand while {len(args)} input!')
        if isinstance(k, (Expr, Index)) and not isinstance(k, (Integer, Slice)):
            if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            return dConv_s((k - dLinspace(0, k - 1)) / k * DT(psi(args[0]), Slice(0, k - 1)),
                           _dtify(args[0], Slice(1, k)))
        elif isinstance(k, Slice):
            if k.start >= 1:
                Psi1 = DT(psi(args[0]), Slice(0, k.end - k.start))
                temp = dConv_v((k.end - dLinspace(0, k.end - k.start)) / k.end * Psi1, _dtify(args[0], k))
                for i in range(k.start):
                    temp = temp + (k.end - dLinspace(k.start - i, k.end - i)) / k.end * \
                           DT(psi(args[0]), Slice(k.start - i, k.end - i)) * _dtify(args[0], i)
                return temp
            else:
                phi_ = DT(phi(args[0]), 0)
                psi_ = DT(psi(args[0]), Slice(0, k.end))
                return dDelta(Slice(0, k.end)) * phi_ + \
                    dConv_v((k.end - dLinspace(0, k.end)) / k.end * psi_,
                            _dtify(args[0], Slice(0, k.end))) * (1 - dDelta(Slice(0, k.end)))
        else:  # integer
            if k > 1:
                return dConv_s((k - dLinspace(0, k - 1)) / k * DT(psi(args[0]), Slice(0, k - 1)),
                               _dtify(args[0], Slice(1, k)))
            elif k == 1:
                return DT(psi(args[0]), 0) * _dtify(args[0], 1)
            elif k == 0:
                if args[0].__repr__() == 't':
                    # sin(t)[0] = 0
                    return 0
                else:
                    return DT(phi(args[0]), 0)
            else:
                raise ValueError(f'DT index must be great than zero!')


@implements_dt_algebra(cos)
class dCos(Function):
    r"""
    This function returns DT of $\psi(t)=\cos(x(t))$, which is denoted by $\phi[k]$.

    For expression $\psi(t)=\cos(x(t))$,

    if the order of DT is an `Index` object $k$, it returns

    .. math::
        \begin{cases}
            -\sum_{m=0}^{k-1} \frac{k-m}{k}\phi[m]x[k-m]=-\left(\frac{k-(0:k-1)}{k}*\phi[0:k-1]\right)\otimes x[1:k]& k\geq 2\\
            -\phi[0]* x[1]& k= 1\\
            \psi(0)& k=0
        \end{cases}

    Else if the order of DT is a `Slice` object $k_0:k_1$, it returns

    .. math::

        \begin{cases}
          -\left(\frac{k_1-(0:k_1-k_0)}{k_1}*\phi[0:k_1-k_0]\right)\overline{\otimes}
          x[k_0:k_1]-\sum_{i=0}^{k_0-1}\left(\frac{k_1-(k_0-i:k_1-i)}{k_1}*\phi[k_0-i:k_1-i]\right)* x[i] & k_0\geq 1 \\
          -\left(\left(\frac{k_1-(0:k_1)}{k_1}*\phi[0:k_1]\right)\overline{\otimes} x[0:k_1]\right)*(1-\delta[0:k_1])-\psi[0]*\delta[0:k_1] & k_0=0
        \end{cases}

    See Also
    ========

    dSin

    """

    @classmethod
    def eval(cls, k, *args):
        if len(args) > 1:
            raise ValueError(f'Sin supports one operand while {len(args)} input!')
        if isinstance(k, (Expr, Index)) and not isinstance(k, (Integer, Slice)):
            if any([not isinstance(symbol, Index) for symbol in k.free_symbols]):
                raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
            return -dConv_s(((k - dLinspace(0, k - 1)) / k) * DT(phi(args[0]), Slice(0, k - 1)),
                            _dtify(args[0], Slice(1, k)))
        elif isinstance(k, Slice):
            if k.start >= 1:
                Phi1 = DT(phi(args[0]), Slice(0, k.end - k.start))
                temp = -dConv_v((k.end - dLinspace(0, k.end - k.start)) / k.end * Phi1, _dtify(args[0], k))
                for i in range(k.start):
                    temp = temp - (k.end - dLinspace(k.start - i, k.end - i)) / k.end * \
                           DT(phi(args[0]), Slice(k.start - i, k.end - i)) * _dtify(args[0], i)
                return temp
            else:
                phi_ = DT(phi(args[0]), Slice(0, k.end))
                psi_ = DT(psi(args[0]), 0)
                return -dDelta(Slice(0, k.end)) * psi_ - \
                    dConv_v((k.end - dLinspace(0, k.end)) / k.end * phi_,
                            _dtify(args[0], Slice(0, k.end))) * (1 - dDelta(Slice(0, k.end)))
        else:  # integer
            if k > 1:
                return -dConv_s((k - dLinspace(0, k - 1)) / k * DT(phi(args[0]), Slice(0, k - 1)),
                                _dtify(args[0], Slice(1, k)))
            elif k == 1:
                return -DT(phi(args[0]), 0) * _dtify(args[0], 1)
            elif k == 0:
                if args[0].__repr__() == 't':
                    # cos(t)[0] = 1
                    return 1
                else:
                    return DT(psi(args[0]), 0)
            else:
                raise ValueError(f'DT index must be great than zero!')


@implements_dt_algebra(Pow)
class dPow(Function):
    """
    This function returns DT of power.
    """
    pass


@implements_dt_algebra(Derivative)
class dDerivative(Function):
    r"""
    This function returns DT of $x'(t)$.

    For expression $x'(t)$, if the order of DT is an `Index` object $k$, it returns

    $$(k+1)*x[k+1]$$
    """

    @classmethod
    def eval(cls, k, *args):
        if args[1][0].name != 't':
            raise ValueError("Support time derivative only!")
        if isinstance(k, (Index, Expr, Slice, Integer)):
            return (k + 1) * _dtify(args[0], k + 1)
        else:
            # args[0] is not Index or Slice.
            raise TypeError(f"Invalid inputs type {args[0].__class__.__name__}.")


class dDelta(Function):
    r"""The Kronecker delta function, $\delta(k)$.

    Explanation
    ===========

    This function will evaluate automatically in the
    case $k$ is some integer number, that is,

    .. math::

        \Delta[k]=
        \begin{cases}
          1 & k=0 \\
          0 & k\in \mathbb{N}^+
        \end{cases}

    """

    @classmethod
    def eval(cls, *args):
        if len(args) > 1:
            raise ValueError("Support one argument only!")
        if isinstance(args[0], Integer):
            # Function class automatically convert int into sp.Integer.
            if args[0] == 0:
                return 1
            else:
                return 0
        if not isinstance(args[0], Index) and not isinstance(args[0], Slice):
            # args[0] is not Index or Slice.
            raise TypeError(f"Invalid inputs type {args[0].__class__.__name__}.")

    def _latex(self, printer):
        k = self.args[0]
        k = printer._print(k)
        return r'\Delta \left [ %s \right ]' % k


class dConv_s(Function):
    r"""
    The convolution of vectors, which returns vectors and will not be evaluated automatically.

    Explanation
    ===========

    The arguments of `dConv_s` are vectors $v_i\in \mathbb{R}^{n+1}\ (i=1,2,3,\cdots)$. Suppose each of $v_i$ denotes
    an $n$-th degree polynomial, `dConv_s` returns the coefficients of the $n$-th degree term of the multiplications
    of these polynomials.

    For example, let $x$ and $y$ be two vectors in $\mathbb{R}^{n+1}$, then `dConv_s(x,y)` returns

    .. math::

        \sum_{m=0}^{n}x[m]y[n-m]

    We denote by $\bigotimes$ the `dConv_s` function. So that `dConv_s(x,y)` can be written as $x[0:n]\bigotimes y[0:n]$
    or $x\bigotimes y$ for brevity.

    """

    is_commutative = False
    is_dConv = True

    def __mul__(self, other):
        args = list(self.args)
        args[-1] = Mul(args[-1] * other)
        return self.func(*args)

    def __rmul__(self, other):
        args = list(self.args)
        args[0] = Mul(other * args[0])
        return self.func(*args)

    @classmethod
    def eval(cls, *args):
        # if the operands are matrices, then
        if all([isinstance(arg, ImmutableDenseMatrix) for arg in args]):
            x_ = Symbol('x_')  # declare temporary symbol x_ to perform polynomial multiplications
            temp = reduce(lambda a, b: a * b,
                          [arg[0] if arg.shape[0] == 1 else arg[0] + arg[1] * x_ for arg in args]).expand()
            if args[0].shape[0] == 1:
                return temp
            else:
                return temp.coeff(x_)

    def _eval_expand_func(self, **hints):
        """
        traverse to flatten dConv_s tree
        """

        args = self.args
        i = -1
        for arg in args:
            i = i + 1
            if arg.has(dConv_v):
                if arg.is_Add and isinstance(arg, Expr):
                    x, y = arg.as_two_terms()
                    temp_args1 = list(args)
                    temp_args1[i] = x
                    temp_args2 = list(args)
                    temp_args2[i] = y
                    return self.func(*temp_args1).expand(**hints) + \
                        self.func(*temp_args2).expand(**hints)
                elif arg.func == dConv_v:
                    # extract the arguments of sub-dConv_s nodes
                    if i == 0:
                        return self.func(*(list(arg.args) + list(args[1:]))).expand(**hints)
                    elif i == len(args) - 1:
                        return self.func(*(list(args[:-1]) + list(arg.args))).expand(**hints)
                    else:
                        return self.func(*(list(args[0:i - 1]) + list(arg.args) + list(args[0:i + 1]))).expand(**hints)
                elif arg.is_Mul:
                    # Replace ``Mul(-1, dConv_s())`` by ``(-1) * dConv_s()`` so that ``__mul__()`` and ``__rmul__()``
                    # are triggered. -1 should be converted to python number.
                    args = []
                    for arg_ in arg.args:
                        if isinstance(arg_, Integer):
                            args = [*args, int(arg_)]
                        elif isinstance(arg_, Float):
                            args = [*args, float(arg_)]
                        else:
                            args = [*args, arg_]
                    arg = reduce(lambda a, b: a * b, args)
                    temp_args = list(self.args)
                    temp_args[i] = arg
                    return self.func(*temp_args).expand(**hints)

        return self

    def _latex(self, printer):

        arg_latex_str = []
        for arg in self.args:
            if isinstance(arg, Symbol):
                arg_latex_str = [*arg_latex_str, printer._print(arg)]
            else:
                arg_latex_str = [*arg_latex_str, r'\left (' + printer._print(arg) + r'\right )']
        _latex_str = arg_latex_str[0]
        for arg_latex_str_ in arg_latex_str[1:]:
            _latex_str = _latex_str + r'\otimes ' + arg_latex_str_
        return _latex_str


class dConv_v(Function):
    r"""
    The vectorization of `dConv_s`, which returns vectors and will not be evaluated automatically.

    Explanation
    ===========

    The arguments of `dConv_v` are vectors $v_i\in \mathbb{R}^{n+1}\ (i=1,2,3,\cdots)$. Suppose each of $v_i$ denotes
    an $n$-th degree polynomial, `dConv_v` returns the coefficients of the $0$-th to $n$-th degree term of the multiplications
    of these polynomials.

    For example, let $x$ and $y$ be two vectors in $\mathbb{R}^{n+1}$, then `dConv_v(x,y)` returns

    .. math::

        \begin{bmatrix}
        x[0]y[0]&\cdots&\sum_{m=0}^{n-1}x[m]y[n-m]&\sum_{m=0}^{n}x[m]y[n-m]
        \end{bmatrix}

    We denote by $\overline{\bigotimes}$ the `dConv_v` function. So that `dConv_v(x,y)` can be written as $x\overline{\bigotimes} y$.

    """

    is_commutative = False
    is_dConv = True

    def __mul__(self, other):
        args = list(self.args)
        args[-1] = Mul(args[-1] * other)
        return self.func(*args)

    def __rmul__(self, other):
        args = list(self.args)
        args[0] = Mul(other * args[0])
        return self.func(*args)

    def _eval_expand_func(self, **hints):
        """
        traverse to flatten dConv_s tree
        """

        args = self.args
        i = -1
        for arg in args:
            i = i + 1
            if arg.has(dConv_v):
                if arg.is_Add and isinstance(arg, Expr):
                    x, y = arg.as_two_terms()
                    temp_args1 = list(args)
                    temp_args1[i] = x
                    temp_args2 = list(args)
                    temp_args2[i] = y
                    return self.func(*temp_args1).expand(**hints) + \
                        self.func(*temp_args2).expand(**hints)
                elif arg.func == dConv_v:
                    # extract the arguments of sub-dConv_s nodes
                    if i == 0:
                        return self.func(*(list(arg.args) + list(args[1:]))).expand(**hints)
                    elif i == len(args) - 1:
                        return self.func(*(list(args[:-1]) + list(arg.args))).expand(**hints)
                    else:
                        return self.func(*(list(args[0:i - 1]) + list(arg.args) + list(args[0:i + 1]))).expand(**hints)
                elif arg.is_Mul:
                    # Replace ``Mul(-1, dConv_v())`` by ``(-1) * dConv_v()`` so that ``__mul__()`` and ``__rmul__()``
                    # are triggered. -1 should be converted to python number.
                    args = []
                    for arg_ in arg.args:
                        if isinstance(arg_, Integer):
                            args = [*args, int(arg_)]
                        elif isinstance(arg_, Float):
                            args = [*args, float(arg_)]
                        else:
                            args = [*args, arg_]
                    arg = reduce(lambda a, b: a * b, args)
                    temp_args = list(self.args)
                    temp_args[i] = arg
                    return self.func(*temp_args).expand(**hints)

        return self

    def _latex(self, printer):

        arg_latex_str = []
        for arg in self.args:
            if isinstance(arg, Symbol):
                arg_latex_str = [*arg_latex_str, printer._print(arg)]
            else:
                arg_latex_str = [*arg_latex_str, r'\left (' + printer._print(arg) + r'\right )']
        _latex_str = arg_latex_str[0]
        for arg_latex_str_ in arg_latex_str[1:]:
            _latex_str = _latex_str + r'\overline{\otimes} ' + arg_latex_str_
        return _latex_str


class dLinspace(Function):
    r"""
    The symbolic linspace function, which will not be evaluated automatically.

    """
    is_commutative = False

    @classmethod
    def eval(cls, m, n):
        if isinstance(m, Slice) or isinstance(n, Slice):
            raise TypeError('Supports only integer input!')

    def _latex(self, printer):
        m, n = self.args
        _m, _n = printer._print(m), printer._print(n)
        return r'\left [ %s : %s \right ]' % (_m, _n)


def extract_unknown_term(expr: Expr, k: [Index, int]):
    r"""
    To extract unknown terms from expressions with :py:class:`dConv_s` ($\bigotimes$) and
    :py:class:`dConv_v` ($\overline{\bigotimes}$) expanded.

    For example, if we have

    .. math::

        z[k]=x[0:k]\otimes y[0:k]

    which we want to incorporate in computation for unknown $x[k]$ and $y[k]$, we have to expand $z[k]$ as

    .. math::

        z[k]=x[1:k-1]\otimes y[1:k-1] + x[0]y[k] + x[k]y[0]

    Examples
    ========

    See :py:func:`dtify`

     Parameters
    ==========

    expr : Expr

        An expression to be evaluated.

    k : Index, int

        DT Index of expr.

    Returns
    =======

    RHS : Expr

        The known parts of input expr.

    LHS: Expr

        The unknown parts of input expr.

    Explanation
    ===========

    Let's introduce the basis methodology by taking DT expression $z[k]=x[0:k]\otimes y[0:k]$ as an example.

    For $k$-th order DT expression, it is obvious that the coefficients of unknown $k$-th order terms are always $0$-th
    order, and vice versa. Hence, we first extract $0$th and $k$th terms from the DT slices, which form a matrix. Here,
    we substitute ``sympy.Matrix`` $[x[0], x[k]]$ and $[y[0], y[k]]$ for $x[0:k]$ and $y[0:k]$
    respectively in the original expression, which derives

    .. math::

        \begin{bmatrix}
            x[0]\\
            x[k]
        \end{bmatrix}
        \otimes
        \begin{bmatrix}
            y[0]\\
            y[k]
        \end{bmatrix}

    The convolution of these two matrices with shape (2,1) are performed by overriding the :py:meth:`dConv_s.eval` method,
    which computes the multiplication of polynomial $x[0]+x[k]*x\_$ and $y[0]+y[k]*x\_$. Then the convolution result is the
    coefficient of $x\_$ term, that is

    .. math::

        x[0]y[k] + x[k]y[0].

    Next, the known terms can be obtained by substituting $x[1:k-1]$ and $y[1:k-1]$ for $x[0:k]$ and $y[0:k]$ in $z[k]$
    respectively, which derives

    .. math::

        x[1:k-1]\otimes y[1:k-1].

    """
    # to ensure that there is no dConv_v in expr
    expr = expr.expand(func=True, mul=False)
    if expr.has(dConv_v):
        raise ValueError(f"Expression contain dCon_v function!")
    RHS = expr
    LHS = -expr
    # derive $x[1:k-1]\otimes y[1:k-1]$, the known terms (right hand side), using sp.subs
    for DT_ in list(RHS.free_symbols):
        if isinstance(DT_, DT):
            if isinstance(DT_.index, Slice):
                start = DT_.index.start
                end = DT_.index.end
                if start == 0:
                    start = 1
                if end == k:
                    end = k - 1
                RHS = RHS.subs(DT_, DT(DT_.symbol, Slice(start, end), commutative=DT_.is_commutative))
            elif DT_.index == k:
                # if there is independent $x[k]$ in expr
                RHS = RHS.subs(DT_, 0)

    # search for dLinspace function and substitute
    dlinspaces = search_for_func(RHS, dLinspace)
    if len(dlinspaces) > 0:
        for dlinspace in dlinspaces:
            start = dlinspace.args[0]
            end = dlinspace.args[1]
            array = []
            if start == 0:
                start = 1
                array += [0]
            if end == k:
                end = k - 1
                array += [k]
            RHS = RHS.subs(dlinspace, dLinspace(start, end))
            if len(array) > 1:
                LHS = LHS.subs(dlinspace, Matrix(array))
            else:
                LHS = LHS.subs(dlinspace, *array)

    # derive $x[0]y[k] + x[k]y[0]$, the unknown terms (left hand side).
    for DT_ in list(LHS.free_symbols):
        if isinstance(DT_, DT):
            if isinstance(DT_.index, Slice):
                start = DT_.index.start
                end = DT_.index.end
                arg_list = []
                if start == 0:
                    arg_list += [DT(DT_.symbol, 0)]
                if end == k:
                    arg_list += [DT(DT_.symbol, k)]
                LHS = LHS.subs(DT_, Matrix(arg_list), commutative=DT_.is_commutative)
            elif DT_.index != k and DT_.index != 0:
                LHS = LHS.subs(DT_, 0)

    # if LHS contains t[k] then move it to RHS
    for symbol_ in list(LHS.free_symbols):
        if isinstance(symbol_, DT):
            if symbol_.symbol_name == 't':
                temp = LHS.coeff(symbol_) * symbol_
                LHS = LHS - temp
                RHS = RHS - temp
    # Eliminate dDelta(k) in LHS
    temp_delta_list = search_for_func(LHS, dDelta)
    for dDelta_ in temp_delta_list:
        if dDelta_.args[0] == k:
            LHS = LHS.subs(dDelta_, 0)

    # If the coefficients of unknown terms contain k, then divide both sides by the unknown coefficients
    # this process combined into the above for-loop because the free_symbols dict of
    # LHS changes accordingly after moving t[k] from LHS to RHS.
    # This case typically happens when there is derivative in the equation, and
    # we use sp.simplify function to perform simplification, but it is not a robust method.
    for symbol_ in list(LHS.free_symbols):
        if isinstance(symbol_, DT):
            if any([isinstance(arg, Index) for arg in list(LHS.coeff(symbol_).free_symbols)]):
                temp = LHS.coeff(symbol_)
                LHS = symbol_
                RHS = simplify(RHS / temp)
    # check if there is still k in LHS, if so, raise error
    for symbol_ in list(LHS.free_symbols):
        if isinstance(symbol_, DT):
            if any([isinstance(arg, Index) for arg in list(LHS.coeff(symbol_).free_symbols)]):
                raise ValueError(f"{k}-th order still found in Left hand side of equation.")

    return RHS, LHS


def search_for_func(expr, func: type[Function]) -> List[Expr]:
    r"""
    Return the instance of given function patterns in a sympy expression

    Examples
    ========

    >>> from Solverz.eqn import Eqn
    >>> from Solverz.sas.sas_alg import dtify, search_for_func
    >>> from sympy import cos
    >>> test = Eqn(name='test', eqn='cos(x)')
    >>> search_for_func(test.expr, cos)
    [cos(x)]
    >>> test_ = dtify(test.expr,etf=True,eut=True)
    >>> test_[1]
    -phi_x[0]*x[k] - psi_x[k]=dConv_s(phi_x[1:k - 1]*(k - dLinspace(1, k - 1))/k, x[1:k - 1])
    >>> search_for_func(test_[1].RHS, dLinspace)
    [dLinspace(1, k - 1)]

    Parameters
    ==========

    expr : Expr or number

        An expression to be evaluated.

    func : function patter to find

    Returns
    =======

    exprs : List[expr]
        List of functions of pattern func.

    """
    results = []
    pt = preorder_traversal(expr)
    for node in pt:
        if isinstance(node, func):
            results += [node]
    return results


class k_eqn:
    """
    The basic class of k-domain equation

    """

    def __init__(self, expr: Expr, eut=True, int_var=None):
        """

        :param expr:
        """

        if not isinstance(expr, Expr):
            raise TypeError(f"Expect sympy.Expr, got {expr.__class__}")
        self.expr = expr
        if not any([isinstance(symbol_, DT) for symbol_ in self.SYMBOLS.values()]):
            raise TypeError("Support dtified equation only.")
        self.k = self.obtain_unknown_index()
        self.eut = eut
        if eut:
            self.RHS, self.LHS = extract_unknown_term(expr, self.k)
        else:
            self.RHS = Integer(0)
            self.LHS = expr

        self.int = dict()  # if the k_eqn stems from some intermediate variable
        if int_var:
            if isinstance(int_var, (psi, phi)):
                if isinstance(int_var, psi):
                    self.int['var'] = int_var
                    self.int['func'] = cos(int_var.eqn)
                else:
                    self.int['var'] = int_var
                    self.int['func'] = sin(int_var.eqn)
            else:
                raise NotImplementedError(f"Unknown intermediate variable {int_var}")

        # separate the coefficient of unknown terms
        self.COEFF = Coeffs()
        for symbol_ in list(self.LHS.free_symbols):
            if isinstance(symbol_, DT):
                self.COEFF.add_coeff(self.LHS.coeff(symbol_), symbol_.symbol_name)

        self.RHS_NUM_FUNC: Callable = lambda x: None  # return empty function

    @property
    def SYMBOLS(self):
        symbol_dict = dict()
        for symbol in list(self.expr.free_symbols):
            symbol_dict[symbol.name] = symbol
        return symbol_dict

    def obtain_unknown_index(self) -> Expr:
        # find the largest Index of DT symbols in the expression
        index = []
        value = []
        for symbol_ in self.SYMBOLS.values():
            if isinstance(symbol_, DT):
                index_ = symbol_.index
                if isinstance(index_, Slice):
                    index_ = index_.end
                index += [index_]
                if isinstance(index_, (Number, int, float)):
                    value += [index_]
                elif isinstance(index_, Expr):
                    value += [lambdify(list(index_.free_symbols), index_)(0)]
        return index[value.index(max(value))]  # index method of list returns the first location of the input arg

    def extract_unknown_terms(self):
        if not self.eut:
            self.RHS, self.LHS = extract_unknown_term(self.expr, self.k)
        else:
            raise ValueError("Unknown terms have already been extracted!")

    def lambdify(self, modules):
        for symbol_ in list(self.k.free_symbols):
            if isinstance(symbol_, Index):
                self.RHS_NUM_FUNC = lambdify(symbol_, pre_lambdify(self.RHS), modules=modules)
        self.COEFF.lambdify(modules)

    def __repr__(self):
        return self.LHS.__repr__() + r"=" + self.RHS.__repr__()

    def _repr_latex_(self):
        """
        So that jupyter notebook can display latex equation of k_eqn object.
        :return:
        """
        return r"$\displaystyle %s$" % (latex(self.LHS) + r"=" + latex(self.RHS))


class Coeffs:
    """
    The class of coeff of unknown DT terms in LHS of k_eqn
    """

    def __init__(self):
        self.expr: Dict[str, Expr] = dict()
        self.NUM_FUNC: Dict[str, Callable] = dict()

    def add_coeff(self, expr: Expr, var: str):
        self.expr[var] = expr

    def lambdify(self, modules):
        for var, expr in self.expr.items():
            self.NUM_FUNC[var] = lambdify([], expr, modules=modules)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.NUM_FUNC[item]()

    def __repr__(self):
        return self.expr.__repr__()


def _dtify(Node: Expr, k: [Index, Slice, Type[Expr], int]):
    """
    Replace sympy operator by DT operators/symbols
    """
    if isinstance(Node, tuple(dfunc_mapping.keys())):
        return dfunc_mapping[Node.func](k, *Node.args)
    elif isinstance(Node, Symbol) and not isinstance(Node, Constant):
        if Node.name != 't':
            return DT(Node, k)
        else:  # dt of t
            if isinstance(k, (Expr, Index)) and not isinstance(k, Slice):
                if all([isinstance(symbol, Index) for symbol in k.free_symbols]) is False:
                    raise TypeError(f"Non-Index symbol found in Index Expression {k}!")
                else:
                    return DT(Node, k)
            elif isinstance(k, Slice):
                return DT(Node, k)
            else:  # integer
                if k < 0:
                    raise ValueError("DT index must be great than zero!")
                elif k == 1:
                    return 1
                else:
                    return 0
    elif isinstance(Node, (Number, Constant)):
        return Node * dDelta(k)
    else:
        raise TypeError(f"Unsupported Expr {Node.func}!")


def dtify(expr, k=None, etf=False, eut=False, constants=None) -> Union[k_eqn, List[k_eqn]]:
    r"""
    Derives DTs of expressions and returns k equations.

    Examples
    ========

    >>> from Solverz.eqn import Eqn
    >>> from Solverz.sas.sas_alg import dtify
    >>> Eq_prime = Eqn(name='Eq_prime', eqn='Eqp-cos(delta)*(Uxg+ra*Ixg-Xdp*Iyg)-sin(delta)*(Uyg+ra*Iyg+Xdp*Ixg)')
    >>> dtify(Eq_prime.expr)
    -Eqp[k] + dConv_s(Uxg[0:k] + dConv_v(Ixg[0:k], ra[0:k]) - dConv_v(Iyg[0:k], Xdp[0:k]), psi_delta[0:k]) + dConv_s(Uyg[0:k] + dConv_v(Ixg[0:k], Xdp[0:k]) + dConv_v(Iyg[0:k], ra[0:k]), phi_delta[0:k])=0

    Parameters
    ==========

    expr : Expr or number

        An expression to be evaluated.

    k : Index, Slice, Type[Expr], int

        DT Index of expr.

    etf : bool

        If set to ``True``, trigonometric components are extracted from expr and forms new equations.
        Otherwise, trigonometric components will not be extracted from expr but will be simply replaced by $\psi$ and
        $\phi$ symbols.

        >>> import sympy as sp
        >>> x, y = sp.symbols('x, y')
        >>> dtify(x * sp.sin(y), etf=True)
        [dConv_s(x[0:k], phi_y[0:k])=0, phi_y[k] - dConv_s(psi_y[0:k - 1]*(k - dLinspace(0, k - 1))/k, y[1:k])=0, psi_y[k] + dConv_s(phi_y[0:k - 1]*(k - dLinspace(0, k - 1))/k, y[1:k])=0]
        >>> dtify(x * sp.sin(y))
        dConv_s(x[0:k], phi_y[0:k])=0

    eut : bool

        If set to ``True``, extract unknown $k$-th terms from the derived DT expressions.

        >>> Eq_test = Eqn(name='Eq_test', eqn='Eqp-cos(delta)*(Uxg+ra*Ixg-Xdp*Iyg)')
        >>> dt_eqn = dtify(Eq_test.expr,etf=True,eut=True, constants=['ra', 'Xdp', 'Eqp'])
        >>> dt_eqn[0]
        Xdp*Iyg[0]*psi_delta[k] + Xdp*Iyg[k]*psi_delta[0] - ra*Ixg[0]*psi_delta[k] - ra*Ixg[k]*psi_delta[0] - Uxg[0]*psi_delta[k] - Uxg[k]*psi_delta[0]=-Eqp*dDelta(k) + dConv_s(-Xdp*Iyg[1:k - 1] + ra*Ixg[1:k - 1] + Uxg[1:k - 1], psi_delta[1:k - 1])
        >>> dt_eqn[1]
        -delta[k]*phi_delta[0] - psi_delta[k]=dConv_s(phi_delta[1:k - 1]*(k - dLinspace(1, k - 1))/k, delta[1:k - 1])
        >>> dt_eqn[2]
        delta[k]*psi_delta[0] - phi_delta[k]=-dConv_s(psi_delta[1:k - 1]*(k - dLinspace(1, k - 1))/k, delta[1:k - 1])

    constants : list of str (variable names)

        For example, if ``x`` is a constant, the DT of x should be ``x*dDelta(k)`` instead of DT object ``x[k]``.

        >>> from Solverz.eqn import Eqn
        >>> Eq_prime = Eqn(name='Eq_prime', eqn='Eqp-cos(delta)*(Uxg+ra*Ixg)')
        >>> dtify(Eq_prime.expr, constants=['ra'])
        -Eqp[k] + dConv_s(ra*Ixg[0:k] + Uxg[0:k], psi_delta[0:k])=0

        In this case, ``ra`` is treated as a constant and no convolution is performed for ``Mul(ra, Ixg)``.

    Returns
    =======

    expr : Expr
        DT expression or list of DT expressions.

    """

    # if node is not instance of sympy.basic, convert node to sympy expressions first.
    # for example, dtify(3) now returns 3*dDelta[k]
    if not isinstance(expr, Basic):
        expr = sympify(expr)

    if constants is not None:
        if not isinstance(constants, list):
            constants = [constants]
        symbol_dict = {}
        for symbol in list(expr.free_symbols):
            symbol_dict[symbol.name] = symbol
        for var_name in constants:
            try:
                expr = expr.subs(symbol_dict[var_name],
                                 Constant(symbol_dict[var_name], commutative=symbol_dict[var_name].is_commutative))
            except KeyError:
                pass

    if any([isinstance(symbol, DT) for symbol in list(expr.free_symbols)]):
        raise TypeError("DT expression cannot be dtified!")
    if k is None:
        k = Index('k')

    if expr.has(sin, cos):
        # subs $\phi$ and $\psi$ for $\sin$ and $\cos$.
        if etf:
            exprs = dict()
            pt = preorder_traversal(expr)
            for node in pt:
                if isinstance(node, sin):
                    phi_ = phi(node.args[0])
                    expr = expr.subs(node, phi_)
                elif isinstance(node, cos):
                    psi_ = psi(node.args[0])
                    expr = expr.subs(node, psi_)
            for symbol in list(expr.free_symbols):
                if isinstance(symbol, psi):
                    exprs[symbol] = symbol - cos(symbol.eqn)
                    exprs[phi(symbol.eqn)] = phi(symbol.eqn) - sin(symbol.eqn)
                if isinstance(symbol, phi):
                    exprs[symbol] = symbol - sin(symbol.eqn)
                    exprs[psi(symbol.eqn)] = psi(symbol.eqn) - cos(symbol.eqn)
            exprs = [expr] + [(key, value) for key, value in exprs.items()]
        else:
            pt = preorder_traversal(expr)
            for node in pt:
                if isinstance(node, sin):
                    phi_ = phi(node.args[0])
                    expr = expr.subs(node, phi_)
                elif isinstance(node, cos):
                    psi_ = psi(node.args[0])
                    expr = expr.subs(node, psi_)
            exprs = [expr]
    else:
        exprs = [expr]

    if len(exprs) > 1:  # sin, cos found in the original expression
        return [k_eqn(_dtify(expr_, k), eut) if isinstance(expr_, Expr) else k_eqn(_dtify(expr_[1], k), eut,
                                                                                   int_var=expr_[0]) for expr_ in exprs]
    else:
        return k_eqn(_dtify(exprs[0], k), eut)


def pre_lambdify(expr: Expr):
    r"""
    Extending the `Slice` and `dLinspace` objects by one. This is because, for example a=[1,2,3], a[0:2] returns [1,2].

    Examples
    ========

    >>> from Solverz.sas.sas_alg import dtify, pre_lambdify
    >>> from Solverz.eqn import Ode
    >>> test = Ode(name='test',eqn='2*(y-cos(t))',diff_var='y')
    >>> a=dtify(test.expr,etf=True, k=Index('k'))
    >>> pre_lambdify(a[1].expr)
    psi_t[k] + dConv_s(phi_t[0:k]*(k - dLinspace(0, k))/k, t[1:k + 1])

    """
    expr_ = expr
    for DT_ in list(expr.free_symbols):
        if isinstance(DT_, DT):
            if isinstance(DT_.index, Slice):
                expr_ = expr_.subs(DT_, DT(DT_.symbol, Slice(DT_.index.start, DT_.index.end + 1)))

    dlinspaces = search_for_func(expr_, dLinspace)

    if len(dlinspaces) > 0:
        for dlinspace in dlinspaces:
            start = dlinspace.args[0]
            end = dlinspace.args[1]
            expr_ = expr_.subs(dlinspace, dLinspace(start, end + 1))

    return expr_
