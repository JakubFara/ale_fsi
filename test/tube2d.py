import sys
sys.path.append('..')
from equations import (
    NavierStokesParameters, SaintVenantParameters,
    navier_stokes_ale, saint_venant, saint_venant_newmark,
    mesh_deformation,
)
from src.petsc_ale_simple import AleContinuous
from src.subdomain_solver import SnesSolver
import dolfin as df
import json


# communicator
comm = df.MPI.comm_world
# the mesh and its subdomains
mesh_file = "data/tube2d/mesh.h5"
# the sizes of the mesh is stored here
sizes_file = "data/tube2d/sizes.json"
# mesh labels od mesh parts
labels_file = "data/tube2d/labels.json"

# load mesh labels from json file
with open(labels_file, 'r') as openfile:
    labels = json.load(openfile)

with open(sizes_file, 'r') as openfile:
    sizes = json.load(openfile)

# we load the mesh and the subdomains
with df.HDF5File(comm, mesh_file, "a") as h5_file:
    # first we need to create an empty mesh
    mesh = df.Mesh(comm)
    # load the data stored in `/mesh` to the mesh
    h5_file.read(mesh, "/mesh", False)
    # the dimension of the mesh
    dim = mesh.geometry().dim()
    # we need to create the empty meshfunctions at first
    cell_marker = df.MeshFunction('size_t', mesh, dim)
    facet_marker = df.MeshFunction('size_t', mesh, dim - 1)
    # we load the data to the subdomains markers
    h5_file.read(cell_marker, "/cell_marker")
    h5_file.read(facet_marker, "/facet_marker")


# Funtion spaces
u_vec_element = df.VectorElement("CG", mesh.ufl_cell(), 2)
v_vec_element = df.VectorElement("CG", mesh.ufl_cell(), 2)
p_element = df.FiniteElement("CG", mesh.ufl_cell(), 1)

mixed_function_space = df.FunctionSpace(mesh, df.MixedElement([
    v_vec_element,
    p_element,
]))

u_function_space = df.FunctionSpace(mesh, u_vec_element)


# inflow expression
inflow_expr = df.Expression(
    ("0.0", "-50*max(t, 0.1)*(pow(x[0], 2) - pow(r, 2))"),
    t=0.001,
    r=sizes["radius"],
    degree=2,
)

# boundaty conditions

zero_vec = df.Constant((0.0, 0.0))

bcs = [
    # fluid inflow Dirichlet BC
    df.DirichletBC(
        mixed_function_space.sub(0),
        inflow_expr,
        facet_marker,
        labels["inflow"]
    ),
    # solid inflow Dirichler BC
    df.DirichletBC(
        mixed_function_space.sub(0),
        zero_vec,
        facet_marker,
        labels["inflow_solid"]
    ),
    # solid inflow Dirichler BC
    df.DirichletBC(
        mixed_function_space.sub(0),
        zero_vec,
        facet_marker,
        labels["outflow_solid"]
    ),
]


mesh_bcs = [
    # inflow zero Dirichlet BC
    df.DirichletBC(
        u_function_space,
        zero_vec,
        facet_marker,
        labels["inflow"]
    ),
    # outflow zero Dirichler BC
    df.DirichletBC(
        u_function_space,
        zero_vec,
        facet_marker,
        labels["outflow"]
    ),
]


bcs_zero = [
    (
        df.DirichletBC(
            mixed_function_space.sub(1), df.Constant((0.0)), facet_marker,
            labels["interface"]
        ),
        "solid"
    )
]


# material parameters
dt = 0.01
fluid_density = 1000
fluid_viscosity = 1.0
solid_density = 1000
solid_elastic_modulus = 5e7

fluid_parameters = NavierStokesParameters(
    dt, fluid_density, fluid_viscosity
)

solid_parameters = SaintVenantParameters(
    dt, solid_density, 100000, solid_elastic_modulus
)

# equations

dx = df.Measure("dx")(
    domain=mesh, subdomain_data=cell_marker
)

dx_fluid = dx(labels["fluid"])
dx_solid = dx(labels["solid"])


w = df.Function(mixed_function_space)
w0 = df.Function(mixed_function_space)
w_ = df.TestFunction(mixed_function_space)

# displacement
# acceleration
u0_solid = df.Function(u_function_space)
u0_fluid = df.Function(u_function_space)
a0 = df.Function(u_function_space)

u00 = df.Function(u_function_space)

[v, p] = df.split(w)
v0 = df.Function(u_function_space)
[v_, p_] = df.split(w_)

fluid_pde = navier_stokes_ale(
    [v, u0_fluid, p],
    [v0, u0_fluid, None],
    [v_, None, p_],
    fluid_parameters,
    dx_fluid,
)

solid_pde = saint_venant_newmark(
    v, u0_solid, v0, a0, v_,
    solid_parameters,
    dx_solid,
)

p_pde_solid = df.inner(df.grad(p), df.grad(p_)) * dx_solid

u_ = df.TestFunction(u_function_space)

ale_mesh_form = mesh_deformation(u0_fluid, u_, dx_fluid, mesh)


solver = AleContinuous(
    fluid_pde, solid_pde + p_pde_solid, w, bcs, bcs_zero=bcs_zero,
    comm=comm,
)


ale_mesh_solver = SnesSolver(
    ale_mesh_form, mesh_bcs, u0_fluid, u0_solid,
    mesh, cell_marker, labels["fluid"],
    comm=comm
)

ale_mesh_solver.set_solver()

gamma = 0.5
betta = 0.25


with df.XDMFFile(comm, "results/tube2d_v.xdmf") as xdmf_v:
    xdmf_v.parameters["flush_output"] = True
    xdmf_v.parameters["functions_share_mesh"] = True

with df.XDMFFile(comm, "results/tube2d_p.xdmf") as xdmf_p:
    xdmf_p.parameters["flush_output"] = True
    xdmf_p.parameters["functions_share_mesh"] = True

with df.XDMFFile(comm, "results/tube2d_u.xdmf") as xdmf_u:
    xdmf_u.parameters["flush_output"] = True
    xdmf_u.parameters["functions_share_mesh"] = True

t = 0
t_end = 1.0

u0_solid_temp = df.Function(u_function_space)
a0_solid_temp = df.Function(u_function_space)
u00 = df.Function(u_function_space)

while t < t_end:
    t += dt
    df.info(f"t = {t}")
    inflow_expr.t = t
    solver.set_solver(params={"snes_": {"max_it": 3}})
    solver.solve()
    for i in range(9):
        df.info(f"it = {i + 1}")
        [v, p] = solver.u.split(True)
        w.assign(solver.u)

        # update the vectors from previous step
        u0_solid_temp.vector()[:] = (
            u0_solid.vector() + dt * v0.vector()
            + (dt)**2 / 2 * ((1 - 2 * betta) * a0.vector())
        )
        a0_solid_temp.vector()[:] = (
            (1 / (gamma * dt))
            * (v.vector() - v0.vector() - (1 - gamma) * dt * a0.vector())
        )
        u0_solid_temp.vector()[:] += (dt)**2 / 2 * (2 * betta * a0_solid_temp.vector()[:])

        u00.assign(u0_fluid)
        ale_mesh_solver.rest_function = u0_solid_temp
        # ale_mesh_solver.set_solver(params={"snes_": {"max_it": 4}})
        df.info("solving ale deformation")
        ale_mesh_solver.solve()
        df.info("end solving ale deformation")
        # xdmf_v.write(v, t + dt / 10 * (i + 1))
        # xdmf_p.write(p, t + dt / 10 * (i + 1))
        # xdmf_u.write(u0_fluid, t + dt / 10 * (i + 1))
        error = (
            df.assemble(df.inner(u0_fluid - u00, u0_fluid - u00)*df.dx)
            / df.assemble(df.inner(u0_fluid, u0_fluid)*df.dx)
        )
        df.info(f"error = {error}")

        solver.set_solver(params={"snes_": {"max_it": 1}})
        df.info("solving fsi")
        solver.solve()
        df.info("end solving fsi")
        if error < 1e-8:
            break
    a0.assign(a0_solid_temp)
    v0.assign(v)
    u0_solid.assign(u0_solid_temp)
    xdmf_v.write(v, t)
    xdmf_p.write(p, t)
    xdmf_u.write(u0_fluid, t)
