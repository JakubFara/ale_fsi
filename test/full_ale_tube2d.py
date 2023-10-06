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


petsc_options = {
    'snes_': {
        'rtol': 1.e-10,
        'atol': 1.e-10,
        'stol': 1.e-10,
        'max_it': 10
    },
    'pc_': {
        'type': 'lu',
        'factor_mat_solver_type': 'mumps'
    },
    'mat_': {
        'mumps_': {
            'cntl_1': 1e-4,
            'icntl_14': 800,
            'icntl_24':1
        }
    },
    'ksp_': {
        'type': 'preonly'
    },
}

# communicator
comm = df.MPI.comm_world
mesh_dir = "data/tube2d/"
# the mesh and its subdomains
mesh_file = f"{mesh_dir}mesh.h5"
# the sizes of the mesh is stored here
sizes_file = f"{mesh_dir}sizes.json"
# mesh labels od mesh parts
labels_file = f"{mesh_dir}labels.json"

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
    u_vec_element,
    p_element,
]))


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
    df.DirichletBC(
        mixed_function_space.sub(1),
        zero_vec,
        facet_marker,
        labels["inflow_solid"]
    ),
    # solid inflow Dirichler BC
    df.DirichletBC(
        mixed_function_space.sub(1),
        zero_vec,
        facet_marker,
        labels["outflow_solid"]
    ),
    # inflow zero Dirichlet BC
    df.DirichletBC(
        mixed_function_space.sub(1),
        zero_vec,
        facet_marker,
        labels["inflow"]
    ),
    # outflow zero Dirichler BC
    df.DirichletBC(
        mixed_function_space.sub(1),
        zero_vec,
        facet_marker,
        labels["outflow"]
    ),
]

bcs_zero = [
    # pressure in solid
    (
        df.DirichletBC(
            mixed_function_space.sub(2), df.Constant((0.0)), facet_marker,
            labels["interface"]
        ),
        "solid"
    ),
    # ale displacement
    (
        df.DirichletBC(
            mixed_function_space.sub(1), df.Constant((0.0, 0.0)), facet_marker,
            labels["interface"]
        ),
        "fluid"
    )
]

# material parameters
dt = 0.001
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

[v, u, p] = df.split(w)
[v0, u0, _] = df.split(w0)
[v_, u_, p_] = df.split(w_)

fluid_pde = navier_stokes_ale(
    [v, u, p],
    [v0, u0, None],
    [v_, None, p_],
    fluid_parameters, dx_fluid,
)

solid_pde = saint_venant(
    [v, u, p],
    [v0, u0, None],
    [v_, u_, p_],
    solid_parameters, dx_solid,
)

p_pde_solid = df.inner(df.grad(p), df.grad(p_)) * dx_solid


ale_mesh_form = mesh_deformation(u, u_, dx_fluid, mesh)


solver = AleContinuous(
    fluid_pde + ale_mesh_form, solid_pde + p_pde_solid, w, bcs,
    bcs_zero=bcs_zero, comm=comm,
)

with df.XDMFFile(comm, "results/full_ale/tube2d/v.xdmf") as xdmf_v:
    xdmf_v.parameters["flush_output"] = True
    xdmf_v.parameters["functions_share_mesh"] = True

with df.XDMFFile(comm, "results/full_ale/tube2d/p.xdmf") as xdmf_p:
    xdmf_p.parameters["flush_output"] = True
    xdmf_p.parameters["functions_share_mesh"] = True

with df.XDMFFile(comm, "results/full_ale/tube2d/u.xdmf") as xdmf_u:
    xdmf_u.parameters["flush_output"] = True
    xdmf_u.parameters["functions_share_mesh"] = True

t = 0
t_end = 1.0

while t < t_end:
    t += dt
    df.info(f"t = {t}")
    inflow_expr.t = t
    solver.set_solver(params=petsc_options)
    solver.solve()
    w0.assign(w)
    [v, u, p] = w.split(True)
    xdmf_v.write(v, t)
    xdmf_p.write(p, t)
    xdmf_u.write(u, t)
