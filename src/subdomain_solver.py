import dolfin as df
import petsc4py
petsc4py.init()
import sys
from petsc4py import PETSc
# from dolfin import assemble
import time
from src.options import DEFAULT_OPTIONS, opts_setup
import numpy as np


class SnesSolver(object):
    def __init__(
            self, form, bcs, u: df.Function, rest_function,
            mesh: df.Mesh, cell_marker: df.MeshFunction, cell_index: int,
            comm=None
        ):
        self.function_space = u.function_space()
        self.u = u
        self.rest_function = rest_function
        self.form = form
        self.bcs = bcs
        self.form_grad = df.derivative(form, self.u)
        self.b = df.PETScVector()  # same as b = PETSc.Vec()
        self.J_mat = df.PETScMatrix()
        self.x = df.as_backend_type(self.u.vector())

        self.mesh = mesh
        self.cell_marker = cell_marker
        self.cell_index = cell_index
        # self.submesh = df.SubMesh(mesh, cell_marker, cell_index)
        self.outer_dofs = self.get_outer_dofs()
        self.global_outer_dofs = self.local_to_global_dofs(self.outer_dofs)
        if comm is None:
            self.comm = u.function_space().mesh().mpi_comm()
        else:
            self.comm = comm

    def get_outer_dofs(self):
        dofmap = self.function_space.dofmap()
        loc_to_glob = dofmap.tabulate_local_to_global_dofs()
        outer_dofs = set()
        for cell in df.cells(self.mesh):
            if self.cell_marker[cell] != self.cell_index:
                dofs = set(dofmap.cell_dofs(cell.index()))
                outer_dofs = outer_dofs.union(dofs)
        for i in range(self.u.vector()[:].size, max(outer_dofs) + 1):
            if i in outer_dofs:
                outer_dofs.remove(i)
        return outer_dofs

    def local_to_global_dofs(self, dofs):
        dofmap = self.function_space.dofmap()
        loc_to_glob = dofmap.tabulate_local_to_global_dofs()
        return [loc_to_glob[dof] for dof in dofs]

    def update_x(self, x):
        """Given a PETSc Vec x, update the storage of our solution function u."""
        x.copy(self.x.vec())
        self.x.update_ghost_values()

    def F(self, snes, x, vec_petsc):
        self.update_x(x)
        vec_dolfin = df.PETScVector(vec_petsc)
        df.assemble(self.form, tensor=vec_dolfin)
        for bc in self.bcs:
            bc.apply(vec_dolfin, self.u.vector())

        values = [self.rest_function.vector()[dof] for dof in self.outer_dofs]

        x.setValues(self.global_outer_dofs, values)
        self.u.vector().vec().setValues(self.global_outer_dofs, values)
        vec_petsc.setValues(
            self.global_outer_dofs,
            [0 for _ in range(len(self.outer_dofs))]
        )

        vec_petsc.assemble()
        self.u.vector().vec().assemble()
        x.assemble()

    def J(self, snes, x, mat_petsc, P):
        self.update_x(x)
        mat_dolfin = df.PETScMatrix(mat_petsc)
        df.assemble(self.form_grad, tensor=mat_dolfin, keep_diagonal=True)
        for bc in self.bcs:
            bc.apply(mat_dolfin)

        n = len(list(self.outer_dofs))
        mat_petsc.zeroRows(list(self.global_outer_dofs), diag=1)
        mat_petsc.assemble()
        return True

    def set_solver(self, params=None):
        self.snes = PETSc.SNES().create(self.comm)
        self.ksp = self.snes.getKSP()
        self.snes.setFunction(self.F, self.b.vec())
        self.snes.setJacobian(self.J, self.J_mat.mat())

        self.snes.computeFunction(self.x.vec(), self.b.vec())
        self.snes.computeJacobian(self.x.vec(), self.J_mat.mat())

        self.snes.setMonitor(SnesMonitor())
        if params is None:
            params = DEFAULT_OPTIONS
        opts_setup(params)
        self.snes.setFromOptions()
        self.ksp.setFromOptions()

    def solve(self):
        self.snes.setUp()
        self.x.vec().assemble()
        self.snes.solve(None, self.x.vec())

    def make_XDMFfiles(self, names, directory):
        self.f = []
        self.names = names
        for name in names:
            with df.XDMFFile(self.comm, "{}/{}.xdmf".format(directory, name) ) as xdmf:
                xdmf.parameters["flush_output"] = True
                xdmf.parameters["functions_share_mesh"] = True
                self.f.append(xdmf)

    def write_to_file(self, t):
        n = 0
        w = self.u.split(True)
        for files in self.f:
            # info("saving to file: ",name)
            w[n].rename(self.names[n], self.names[n])  #change name of function
            files.write(w[n], t)
            n += 1


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
