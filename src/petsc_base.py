import petsc4py
petsc4py.init()
import sys
from petsc4py import PETSc
import dolfin as df
from src.options import DEFAULT_OPTIONS, opts_setup
# from ale.tools.form import assemble

from dolfin import (
    TrialFunction, derivative, PETScMatrix, PETScVector, MPI, as_backend_type,
    assign, XDMFFile
)



class SnesProblem(df.NonlinearProblem):
    def __init__(self, F1, F2, u, bcs, comm, *args, **kwargs):
        df.NonlinearProblem.__init__(self)
        self.V = u.function_space()
        self.L1 = F1
        self.L2 = F2
        self.a1 = df.derivative(F1, u)
        self.a2 = df.derivative(F2, u)
        self.bcs = bcs
        self.u = u
        self.comm = comm
        self.form_compiler_parameters = {"optimize": True}
        if "form_compiler_parameters" in kwargs:
            self.form_compiler_parameters = kwargs["form_compiler_parameters"]
        self.b = df.PETScVector(self.comm)  # same as b = PETSc.Vec()
        self.J_mat = df.PETScMatrix(self.comm)
        self.x = df.as_backend_type(self.u.vector())
        self.x_petsc = self.x.vec()

        A_dolfin = df.PETScMatrix(self.comm)
        # assemble(self.a1 + self.a2, comm, tensor=A_dolfin, keep_diagonal=True)
        df.assemble(self.a1 + self.a2, tensor=A_dolfin, keep_diagonal=True)
        self.A_petsc = A_dolfin.mat()
        self.xx = self.A_petsc.createVecRight()
        self.xx.axpy(1.0, self.x_petsc)

        self.b_petsc = self.A_petsc.createVecLeft()

    def update_x(self, x):
        """
        Given a PETSc Vec x, update the storage of our solution function u.
        """
        x.copy(self.x_petsc)
        self.x.update_ghost_values()

    def F(self, snes, x, F):
        # Residuum: This function has to be rewritten!
        pass

    def J(self, snes, x, J, P):
        # Jacobian: This function has to be rewritten!
        pass

    def set_solver(self, params=None):
        self.snes = PETSc.SNES().create(self.comm)
        self.ksp = self.snes.getKSP()

        self.snes.setFunction(self.F, self.b_petsc)
        self.snes.setJacobian(self.J, self.A_petsc)

        self.snes.computeFunction(self.xx, self.b_petsc)
        self.snes.computeJacobian(self.xx, self.A_petsc)
        self.snes.setMonitor(SnesMonitor())
        if params is None:
            params = DEFAULT_OPTIONS
        opts_setup(params)
        self.snes.setFromOptions()
        self.ksp.setFromOptions()

    # def set_solver(self, max_it=10, rtol=1.e-10, atol=1.e-10, zero_vec=False,
    #                petsc_opts=None):
    #     self.snes = PETSc.SNES().create(self.comm)
    #     self.opts = PETSc.Options()
    #     if petsc_opts is None:
    #         self.opts["pc_factor_mat_solver_type"] = 'mumps'
    #         self.opts["mat_mumps_cntl_1"] = 1e-4
    #         self.opts["mat_mumps_icntl_14"] = 100
    #         self.opts["mat_mumps_icntl_24"] = 1
    #     else:
    #         for opt_key, opt in petsc_opts:
    #             self.opts[opt_key] = opt

    #     self.snes.setFunction(self.F, self.b_petsc)
    #     self.snes.setJacobian(self.J, self.A_petsc)
    #     if zero_vec is True:
    #         self.xx.zeroEntries()
    #         self.x_petsc.zeroEntries()
    #     self.snes.setSolution(self.xx)
    #     self.snes.computeFunction(self.xx, self.b_petsc)
    #     self.snes.computeJacobian(self.xx, self.A_petsc)
    #     self.snes.setTolerances(
    #         rtol=rtol, atol=atol, stol=1.e-20, max_it=max_it)
    #     self.snes.setMonitor(SnesMonitor())
    #     self.ksp = self.snes.getKSP()
    #     self.ksp.setType('preonly')
    #     self.pc = self.ksp.getPC()
    #     self.pc.setType('lu')
    #     self.pc.setFactorSolverType("mumps")
    #     self.snes.setFromOptions()

    def solve(self):
        # self.snes.setFromOptions()
        self.snes.setUp()
        self.snes.setSolution(self.xx)
        self.snes.solve(None, self.xx)


# ---------------------------MONITORS------------------------------------------
class SnesMonitor():
    def __init__(self):
        self.init = True
    def __call__(self, snes, its, rnorm, *args, **kwargs):
        if self.init is True:
            s = ('%6s' % "it") + (' %5s '% '|') + (' %10s ' % "rnorm")
            PETSc.Sys.Print(s)
            s = ("_____________________________________________________")
            PETSc.Sys.Print(s)

            self.init = False
        s = ('%6d' % its) + (' %5s '% '|') + (' %12.2e' % rnorm)
        PETSc.Sys.Print(s)
        xnorm=snes.vec_sol.norm(PETSc.NormType.NORM_2)
        ynorm=snes.vec_upd.norm(PETSc.NormType.NORM_2)
        iterating = (snes.callConvergenceTest(its, xnorm, ynorm, snes.norm) == 0)
        if not iterating:
            self.init = True
            PETSc.Sys.Print("Convergence reason: {0}  "\
                "  #iterations = {1}".format(snes.reason,its))
            s = ("_____________________________________________________")
            PETSc.Sys.Print(s)
