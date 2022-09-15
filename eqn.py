from __future__ import annotations

from typing import Union, Optional, List
from var import Var
from param import Param
from algebra import Sympify_Mapping
from solverz_array import SolverzArray, Lambdify_Mapping
from sympy import sympify, lambdify, symbols
import warnings


class Eqn:
    """
    The Equation object
    """

    def __init__(self,
                 name: Union[str, List[str]],
                 e_str: Union[str, List[str]],
                 var: Union[List[Var], Var],
                 commutative: Union[list[bool], bool],
                 param: Union[List[Param], Param] = None):

        if isinstance(name, str):
            self.name = [name]
        else:
            self.name = name

        if isinstance(e_str, str):
            self.e_str = [e_str]
        else:
            self.e_str = e_str

        if isinstance(commutative, bool):
            self.commutative = [commutative]
        else:
            self.commutative = commutative

        if len(self.name) > len(self.e_str):
            raise ValueError("Equation names are more than equations!")
        elif len(self.name) < len(self.e_str):
            raise ValueError("Equation names are fewer than equations!")
        else:
            self.free_symbols = set()
            self.EQN = dict()
            for i in range(0, len(self.name)):
                self.EQN[self.name[i]] = sympify(self.e_str[i], locals=Sympify_Mapping)
                self.free_symbols = self.free_symbols.union(self.EQN[self.name[i]].free_symbols, self.free_symbols)
                if not self.commutative[i]:
                    temp_sympify_mapping = Sympify_Mapping
                    for symbol in self.EQN[self.name[i]].free_symbols:
                        temp_sympify_mapping[symbol.name] = symbols(symbol.name, commutative=False)
                    self.EQN[self.name[i]] = sympify(self.e_str[i], locals=temp_sympify_mapping)

        if isinstance(var, Var):
            var = [var]

        self.VAR = {}
        for var_ in var:
            self.VAR[var_.name] = var_

        if param:
            if isinstance(param, Param):
                param = [param]

            self.PARAM = {}
            for param_ in param:
                self.PARAM[param_.name] = param_

        self.check_if_var_and_param_defined()

    def check_if_var_and_param_defined(self):
        for symbol in list(self.free_symbols):
            if symbol.name not in self.VAR and symbol.name not in self.PARAM:
                warnings.warn(f'{symbol.name} not defined')

    def add(self,
            eqn: Union[list, str, Eqn] = None,
            variable: Var = None,
            parameter: Param = None):
        # TODO:  check vars/params with the same names
        pass

    def __add__(self, other):
        # TODO OR NOT TODO. THIS IS A QUESTION.
        pass

    def __repr__(self):
        return f"Eqn {self.name}"
