.. _reference:

=============
API Reference
=============

Functions
---------

.. autoclass:: Solverz.sym_algebra.functions.sin

.. autoclass:: Solverz.sym_algebra.functions.cos

.. autoclass:: Solverz.sym_algebra.functions.exp

.. autoclass:: Solverz.sym_algebra.functions.Abs

.. autoclass:: Solverz.sym_algebra.functions.Sign

.. autoclass:: Solverz.sym_algebra.functions.AntiWindUp

.. autoclass:: Solverz.sym_algebra.functions.Min

.. autoclass:: Solverz.sym_algebra.functions.Saturation

Solvers
-------

AE solver
=========

.. autofunction:: Solverz.solvers.nlaesolver.nr.nr_method

.. autofunction:: Solverz.solvers.nlaesolver.cnr.continuous_nr

.. autofunction:: Solverz.solvers.nlaesolver.lm.lm

FDAE solver
===========

.. autofunction:: Solverz.solvers.fdesolver.fdae_solver

DAE solver
==========

.. autofunction:: Solverz.solvers.daesolver.beuler.backward_euler

.. autofunction:: Solverz.solvers.daesolver.trapezoidal.implicit_trapezoid

.. autofunction:: Solverz.solvers.daesolver.rodas.Rodas
