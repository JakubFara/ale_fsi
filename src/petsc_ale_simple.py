from src.petsc_base import SnesProblem
import dolfin as df
# from dolfin import assemble
import time


class AleContinuous(SnesProblem):
    def __init__(
            self, F_fluid, F_solid, u, bcs,
            bcs_zero=None, comm=None, *args, **kwargs
    ):
        super().__init__(F_fluid, F_solid, u, bcs, comm, *args, **kwargs)
        # pressure and displacement
        if bcs_zero is None:
            bcs_zero = []
        else:
            self.bcs_zero = bcs_zero

    def F(self, snes, x, F):
        self.update_x(x)
        F_fluid = df.Vector()
        F_solid = df.Vector()
        # assemble(self.L2, self.comm, tensor=F2)
        # assemble(self.L1, self.comm, tensor=F1)

        df.assemble(self.L1, tensor=F_fluid)  # L1 - fluid
        df.assemble(self.L2, tensor=F_solid)  # L2 - solid
        # ALE mapping has no contribution on interface

        for bc, where in self.bcs_zero:
            if where == "solid":
                bc.apply(F_solid)
            elif where == "fluid":
                bc.apply(F_fluid)
            elif where == "all":
                bc.apply(F_solid)
                bc.apply(F_fluid)
            else:
                Warning(f"in bcs_zero choose solid/fluid/all  not {where}!")
        # pressure in solid has no contribution on interface
        F_ = df.PETScVector(F)
        F_.zero()
        F_.axpy(1, F_fluid)
        F_.axpy(1, F_solid)
        for bc in self.bcs:
            bc.apply(F_, self.x)

    def J(self, snes, x, J, P):
        self.update_x(x)
        J_ = df.PETScMatrix(J)
        J_.zero()
        J_.apply('insert')
        A1_dolfin = df.PETScMatrix(self.comm)
        A2_dolfin = df.PETScMatrix(self.comm)
        # assemble(self.a1, self.comm, tensor=A1_dolfin, keep_diagonal=True)
        # assemble(self.a2, self.comm, tensor=A2_dolfin, keep_diagonal=True)
        df.assemble(self.a1, tensor=A1_dolfin, keep_diagonal=True)
        df.assemble(self.a2, tensor=A2_dolfin, keep_diagonal=True)

        # ALE mapping has no contribution on interface
        # self.bcs_mesh[0].zero(A2_dolfin)
        for bc, where in self.bcs_zero:
            if where == "solid":
                bc.zero(A2_dolfin)
            elif where == "fluid":
                bc.zero(A1_dolfin)
            elif where == "all":
                bc.zero(A1_dolfin)
                bc.zero(A2_dolfin)
            else:
                Warning(f"in bcs_zero choose solid/fluid/all  not {where}!")
        # pressure in solid has no contribution on interface

        J_.axpy(1, A1_dolfin, False)
        J_.axpy(1, A2_dolfin, False)

        [bc.apply(J_) for bc in self.bcs]

        J.assemble()
        return True
