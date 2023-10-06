from petsc4py import PETSc


def opts_setup(opts_dict):
    opts = PETSc.Options()
    for key, value in opts_dict.items():
        if isinstance(value, dict):
            opts.prefixPush(key)
            opts_setup(value)
            opts.prefixPop()
        else:
            opts[key] = value

DEFAULT_OPTIONS = {
    'ts_': {
        'time_step': 1e-3,
        'max_time': 2.0,
        'type': 'bdf',
        'rtol': 1e-3,
        'atol': 1e-3,
        'time': 0.0,
        'max_steps': 10000,
        'max_step_reject': 10000,
        'max_snes_failures': -1,
        'exact_final_time': 'stepover'
    },
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
            'cntl_1': 1e-8,
            'icntl_14': 100,
            'icntl_24':1
        }
    },
    'ksp_': {
        'type': 'preonly'
    },
    # 'ts_adapt_': {
    #     'type': 'dsp',
    #     'dsp_filter': 'H211PI',
    #     'dt_min':1e-6,
    #     'dt_max': 1e-1,
    #     'clip': '0.3,2.0',
    #     'safety': 0.95,
    #     'reject_safety': 0.8,
    #     'scale_solve_failed': 0.5,
    #     'max_ignore': 1e-6,
    # }
}
"""
                                        TS
Options Database Keys:
 -ts_type <type> - TSEULER, TSBEULER, TSSUNDIALS, TSPSEUDO, TSCN, TSRK, TSTHETA, TSALPHA, TSGLLE, TSSSP, TSGLEE, TSBSYMP, TSIRK
 -ts_save_trajectory - checkpoint the solution at each time-step
 -ts_max_time <time> - maximum time to compute to
 -ts_time_span <t0,...tf> - sets the time span, solutions are computed and stored for each indicated time
 -ts_max_steps <steps> - maximum number of time-steps to take
 -ts_init_time <time> - initial time to start computation
 -ts_final_time <time> - final time to compute to (deprecated: use -ts_max_time)
 -ts_dt <dt> - initial time step
 -ts_exact_final_time <stepover,interpolate,matchstep> - whether to stop at the exact given final time and how to compute the solution at that time
 -ts_max_snes_failures <maxfailures> - Maximum number of nonlinear solve failures allowed
 -ts_max_reject <maxrejects> - Maximum number of step rejections before step fails
 -ts_error_if_step_fails <true,false> - Error if no step succeeds
 -ts_rtol <rtol> - relative tolerance for local truncation error
 -ts_atol <atol> - Absolute tolerance for local truncation error
 -ts_rhs_jacobian_test_mult -mat_shell_test_mult_view - test the Jacobian at each iteration against finite difference with RHS function
 -ts_rhs_jacobian_test_mult_transpose -mat_shell_test_mult_transpose_view - test the Jacobian at each iteration against finite difference with RHS function
 -ts_adjoint_solve <yes,no> - After solving the ODE/DAE solve the adjoint problem (requires -ts_save_trajectory)
 -ts_fd_color - Use finite differences with coloring to compute IJacobian
 -ts_monitor - print information at each timestep
 -ts_monitor_cancel - Cancel all monitors
 -ts_monitor_lg_solution - Monitor solution graphically
 -ts_monitor_lg_error - Monitor error graphically
 -ts_monitor_error - Monitors norm of error
 -ts_monitor_lg_timestep - Monitor timestep size graphically
 -ts_monitor_lg_timestep_log - Monitor log timestep size graphically
 -ts_monitor_lg_snes_iterations - Monitor number nonlinear iterations for each timestep graphically
 -ts_monitor_lg_ksp_iterations - Monitor number nonlinear iterations for each timestep graphically
 -ts_monitor_sp_eig - Monitor eigenvalues of linearized operator graphically
 -ts_monitor_draw_solution - Monitor solution graphically
 -ts_monitor_draw_solution_phase  <xleft,yleft,xright,yright> - Monitor solution graphically with phase diagram, requires problem with exactly 2 degrees of freedom
 -ts_monitor_draw_error - Monitor error graphically, requires use to have provided TSSetSolutionFunction()
 -ts_monitor_solution [ascii binary draw][:filename][:viewerformat] - monitors the solution at each timestep
 -ts_monitor_solution_vtk <filename.vts,filename.vtu> - Save each time step to a binary file, use filename-%%03" PetscInt_FMT ".vts (filename-%%03" PetscInt_FMT ".vtu)
 -ts_monitor_envelope - determine maximum and minimum value of each component of the solution over the solution time

                                SNES
Options Database Keys:
-snes_type <type> - newtonls, newtontr, ngmres, ncg, nrichardson, qn, vi, fas, SNESType for complete list
-snes_stol - convergence tolerance in terms of the norm
                 of the change in the solution between steps
-snes_atol <abstol> - absolute tolerance of residual norm
-snes_rtol <rtol> - relative decrease in tolerance norm from initial
-snes_divergence_tolerance <divtol> - if the residual goes above divtol*rnorm0, exit with divergence
-snes_force_iteration <force> - force SNESSolve() to take at least one iteration
-snes_max_it <max_it> - maximum number of iterations
-snes_max_funcs <max_funcs> - maximum number of function evaluations
-snes_max_fail <max_fail> - maximum number of line search failures allowed before stopping, default is none
-snes_max_linear_solve_fail - number of linear solver failures before SNESSolve() stops
-snes_lag_preconditioner <lag> - how often preconditioner is rebuilt (use -1 to never rebuild)
-snes_lag_preconditioner_persists <true,false> - retains the -snes_lag_preconditioner information across multiple SNESSolve()
-snes_lag_jacobian <lag> - how often Jacobian is rebuilt (use -1 to never rebuild)
-snes_lag_jacobian_persists <true,false> - retains the -snes_lag_jacobian information across multiple SNESSolve()
-snes_trtol <trtol> - trust region tolerance
-snes_convergence_test - <default,skip,correct_pressure> convergence test in nonlinear solver.
                                default SNESConvergedDefault(). skip SNESConvergedSkip() means continue iterating until max_it or some other criterion is reached, saving expense
                                of convergence test. correct_pressure SNESConvergedCorrectPressure() has special handling of a pressure null space.
-snes_monitor [ascii][:filename][:viewer format] - prints residual norm at each iteration. if no filename given prints to stdout
-snes_monitor_solution [ascii binary draw][:filename][:viewer format] - plots solution at each iteration
-snes_monitor_residual [ascii binary draw][:filename][:viewer format] - plots residual (not its norm) at each iteration
-snes_monitor_solution_update [ascii binary draw][:filename][:viewer format] - plots update to solution at each iteration
-snes_monitor_lg_residualnorm - plots residual norm at each iteration
-snes_monitor_lg_range - plots residual norm at each iteration
-snes_monitor_pause_final - Pauses all monitor drawing after the solver ends
-snes_fd - use finite differences to compute Jacobian; very slow, only for testing
-snes_fd_color - use finite differences with coloring to compute Jacobian
-snes_mf_ksp_monitor - if using matrix-free multiply then print h at each KSP iteration
-snes_converged_reason - print the reason for convergence/divergence after each solve
-npc_snes_type <type> - the SNES type to use as a nonlinear preconditioner
-snes_test_jacobian <optional threshold> - compare the user provided Jacobian with one computed via finite differences to check for errors.  If a threshold is given, display only those entries whose difference is greater than the threshold.
   -snes_test_jacobian_view - display the user provided Jacobian, the finite difference Jacobian and the difference between them to help users detect the location of errors in the user provided Jacobian.

Options Database for Eisenstat-Walker method:
-snes_ksp_ew - use Eisenstat-Walker method for determining linear system convergence
-snes_ksp_ew_version ver - version of  Eisenstat-Walker method
-snes_ksp_ew_rtol0 <rtol0> - Sets rtol0
-snes_ksp_ew_rtolmax <rtolmax> - Sets rtolmax
-snes_ksp_ew_gamma <gamma> - Sets gamma
-snes_ksp_ew_alpha <alpha> - Sets alpha
-snes_ksp_ew_alpha2 <alpha2> - Sets alpha2
-snes_ksp_ew_threshold <threshold> - Sets threshold

                                KSP
the Krylov space context

Options Database Keys:
-ksp_max_it - maximum number of linear iterations
-ksp_rtol rtol - relative tolerance used in default determination of convergence, i.e.
                if residual norm decreases by this factor than convergence is declared
-ksp_atol abstol - absolute tolerance used in default convergence test, i.e. if residual
                norm is less than this then convergence is declared
-ksp_divtol tol - if residual norm increases by this factor than divergence is declared
-ksp_converged_use_initial_residual_norm - see KSPConvergedDefaultSetUIRNorm()
-ksp_converged_use_min_initial_residual_norm - see KSPConvergedDefaultSetUMIRNorm()
-ksp_converged_maxits - see KSPConvergedDefaultSetConvergedMaxits()
-ksp_norm_type - none - skip norms used in convergence tests (useful only when not using
                        convergence test (say you always want to run with 5 iterations) to
                       save on communication overhead
                     preconditioned - default for left preconditioning
                     unpreconditioned - see KSPSetNormType()
                     natural - see KSPSetNormType()
-ksp_check_norm_iteration it - do not compute residual norm until iteration number it (does compute at 0th iteration)
        works only for PCBCGS, PCIBCGS and and PCCG
-ksp_lag_norm - compute the norm of the residual for the ith iteration on the i+1 iteration; this means that one can use
        the norm of the residual for convergence test WITHOUT an extra MPI_Allreduce() limiting global synchronizations.
        This will require 1 more iteration of the solver than usual.
-ksp_guess_type - Type of initial guess generator for repeated linear solves
-ksp_fischer_guess <model,size> - uses the Fischer initial guess generator for repeated linear solves
-ksp_constant_null_space - assume the operator (matrix) has the constant vector in its null space
-ksp_test_null_space - tests the null space set with MatSetNullSpace() to see if it truly is a null space
-ksp_knoll - compute initial guess by applying the preconditioner to the right hand side
-ksp_monitor_cancel - cancel all previous convergene monitor routines set
-ksp_monitor - print residual norm at each iteration
-ksp_monitor draw::draw_lg - plot residual norm at each iteration
-ksp_monitor_true_residual - print true residual norm at each iteration
-all_ksp_monitor <optional filename> - print residual norm at each iteration for ALL KSP solves, regardless of their prefix. This is
                                            useful for PCFIELDSPLIT, PCMG, etc that have inner solvers and you wish to track the convergence of all the solvers
-ksp_monitor_solution [ascii binary or draw][:filename][:format option] - plot solution at each iteration
-ksp_monitor_singular_value - monitor extreme singular values at each iteration
-ksp_converged_reason - view the convergence state at the end of the solve
-ksp_use_explicittranspose - transpose the system explicitly in KSPSolveTranspose
-ksp_error_if_not_converged - stop the program as soon as an error is detected in a KSPSolve(), KSP_DIVERGED_ITS is not treated as an error on inner KSPSolves
-ksp_converged_rate - view the convergence rate at the end of the solve

 """
