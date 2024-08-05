# using CUDA: has_cuda_gpu

using Oceananigans
using Oceananigans.AbstractOperations: @at, ∂x, ∂y, ∂z
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Advection: div_Uc
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ
using Oceananigans.TurbulenceClosures: immersed_∇_dot_qᶜ

#import Oceananigans.TurbulenceClosures: viscosity, diffusivity
#using Oceananigans.Fields: @compute 
#using Oceanostics

function get_budget_outputs_tuple(model; )

    # Output: primitive fields + computations
    b = model.tracers.b
    B̄ = model.background_fields.tracers.b
    B = B̄ + b # total buoyancy field
    Cg = model.tracers.Cg

    advection = model.advection

    diffusivities = model.diffusivity_fields

    c_immersed_bc = model.tracers.Cg.boundary_conditions.immersed
    b_immersed_bc = model.tracers.b.boundary_conditions.immersed

    velocities = model.velocities    

    ########## ADVECTIVE FLUX DIVERGENCE
    udiv_c = KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid; computed_dependencies=(velocities, Cg), parameters = (advection,))
   #maybe this instead?
    udiv_c = KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid; computed_dependencies=(advection, velocities, Cg))

    udiv_B = KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid; computed_dependencies=(velocities, B), parameters = (advection,))

    ######### DIFFUSIVE TERMS
    ∇κ∇Cg = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, model.grid, model.closure, diffusivities, val_tracer_index, Cg, model.clock, model_fields, model.buoyancy)
    ∇κ∇Cg_im = KernelFunctionOperation{Center, Center, Center}(immersed_∇_dot_qᶜ, model.grid, Cg, c_immersed_bc, model.closure, diffusivities, val_tracer_index, model.clock, model_fields)
    ∇κ∇B = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, model.grid, model.closure, diffusivities, val_tracer_index, B, model.clock, model_fields, model.buoyancy)
    ∇κ∇B_im = KernelFunctionOperation{Center, Center, Center}(immersed_∇_dot_qᶜ, model.grid, B, b_immersed_bc, model.closure, diffusivities, val_tracer_index, model.clock, model_fields)

    ## ∇c TERMS
    dCdx = Field(@at (Center, Center, Center) ∂x(Cg))
    dCdy = Field(@at (Center, Center, Center) ∂y(Cg))
    dCdz = Field(@at (Center, Center, Center) ∂z(Cg))

    ## ∇b TERMS
    dBdz = Field(@at (Center, Center, Center) ∂z(b))
    dBdx = Field(@at (Center, Center, Center) ∂x(b))
    dBdy = Field(@at (Center, Center, Center) ∂y(b))

    cb = Field(@at (Center, Center, Center) Cg * b)

    dxκdC = Field( ∂x(κ * dCdx))
    dyκdC = Field( ∂y(κ * dCdy))
    dzκdC = Field( ∂z(κ * dCdz))

    dxκdB = Field( ∂x(κ * dBdx))
    dyκdB = Field( ∂y(κ * dBdy))
    dzκdB = Field( ∂z(κ * dBdz))

    ∇κ∇B = Field(@at (Center, Center, Center) dxκdB + dyκdB + dzκdB)

    ∇κ∇Cg = Field(@at (Center, Center, Center) dxκdC + dyκdC + dzκdC)

    outputs = merge(model.tracers,(; ∇κ∇Cg=∇κ∇Cg, ∇κ∇B=∇κ∇B, udiv_c = udiv_c, udiv_B=udiv_B, ))
    

    return outputs

end
