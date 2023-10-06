import dolfin as df
from dataclasses import dataclass


@dataclass
class NavierStokesParameters:
    """
    Parameters for Navier-stokes equation
    """
    dt: float  # time-step
    rho: float  # density
    mu: float  # viscosity


@dataclass
class NeoHookeanParameters:
    """
    Parameters for NeoHookean material
    """
    dt: float  # time-step
    rho: float  # density
    g: float  # elastic modulus


@dataclass
class SaintVenantParameters:
    dt: float  # time-step
    rho: float  # density
    nu0: float  # elastic modulus
    nu1: float


def grad_x(func, def_grad):
    return df.grad(func) * df.inv(def_grad)


def div_x(func, def_grad):
    return df.tr(grad_x(func, def_grad))


# deformation of the fluid-part domain
def mesh_deformation(u, u_, dx, mesh):
    gamma = 9.0 / 8.0
    h = df.CellVolume(mesh) ** (gamma)
    E = df.Constant(10.0) / h
    nu = df.Constant(-0.02)
    mu = E * (2 * (1.0 + nu))
    l = (nu * E) / ((1 + nu) * 1 - 2 * nu)
    return (
        df.inner(mu * 2 * df.sym(df.grad(u)), df.grad(u_)) * dx
         + l * df.inner(df.div(u), df.div(u_)) * dx
    )


def navier_stokes_ale(
    w: list,
    w0: list,
    w_: list,
    parameters: NavierStokesParameters,
    dx: df.Measure,
):
    (v, u, p) = w
    (v_, u_, p_) = w_
    (v0, u0, _) = w0
    k = 1.0 / parameters.dt
    rho = parameters.rho
    gdim = len(v)
    identity = df.Identity(gdim)
    # deformation gradient
    def_grad = identity + df.grad(u)
    cauchy_stress = (
        2 * parameters.mu * df.sym(grad_x(v, def_grad))
        - p * identity
    )
    determinant = df.det(def_grad)
    return (
        (rho * df.inner(k * (v - v0), v_)) * determinant * dx
        + rho * df.inner(grad_x(v, def_grad) * (v - k * (u - u0)), v_)
        * determinant * dx
        + df.inner(cauchy_stress, grad_x(v_, def_grad)) * determinant * dx
        + df.inner(div_x(v, def_grad), p_) * determinant * dx
    )


def saint_venant(
    w: list,
    w0: list,
    w_: list,
    parameters: SaintVenantParameters,
    dx: df.Measure,
):
    (v, u, p) = w
    (v_, u_, p_) = w_
    (v0, u0, _) = w0
    k = 1.0 / parameters.dt
    rho = parameters.rho
    gdim = len(v)
    identity = df.Identity(gdim)
    # deformation gradient
    def_grad = identity + df.grad(u)
    C = (def_grad.T) * def_grad
    nu0 = parameters.nu0
    nu1 = parameters.nu1
    piola = def_grad * (
        0.5 * nu0 * df.tr(C - identity) * identity + nu1 * (C - identity)
    )
    return (
        rho * df.inner(k * (v - v0), v_) * dx
        + df.inner(piola, df.grad(v_)) * dx
        + df.inner(k * (u - u0) - v, u_) * dx
    )


def saint_venant_newmark(
        v, u0, v0, a0, v_,
        parameters: SaintVenantParameters,
        dx: df.Measure,
        gamma=0.5,
        betta=0.25,
):

    k = 1.0 / parameters.dt
    dt = parameters.dt
    a = (k / gamma) * (v - v0 - (1 - gamma) * dt * a0)
    u = u0 + dt * v0 + (dt)**2 / 2 * ((1 - 2 * betta) * a0 + 2 * betta * a)
    rho = parameters.rho
    gdim = len(v)
    identity = df.Identity(gdim)
    # deformation gradient
    def_grad = identity + df.grad(u)
    C = (def_grad.T) * def_grad
    nu0 = parameters.nu0
    nu1 = parameters.nu1
    piola = def_grad * (
        0.5 * nu0 * df.tr(C - identity) * identity + nu1 * (C - identity)
    )
    return (
        rho * df.inner(a, v_) * dx
        + df.inner(piola, df.grad(v_)) * dx
    )


def neo_hookean(
    w: list,
    w0: list,
    w_: list,
    parameters: NeoHookeanParameters,
    dx: df.Measure,
):
    (v, u, p) = w
    (v_, u_, p_) = w_
    (v0, u0, _) = w0
    k = 1.0 / parameters.dt
    rho = parameters.rho
    gdim = len(v)
    identity = df.Identity(gdim)
    # deformation gradient
    def_grad = identity + df.grad(u)
    piola = df.det(def_grad) * (-p * df.inv(def_grad).T + 2 * parameters.g * def_grad)

    determinant = df.det(def_grad)
    return (
        rho * df.inner(k * (v - v0), v_) * dx
        + df.inner(piola, df.grad(v_)) * dx
        + df.inner(k * (u - u0), u_) * dx
        - df.inner(v, u_) * dx
        + df.inner(determinant - 1, p_) * dx
    )
