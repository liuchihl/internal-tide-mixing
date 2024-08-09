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
using Oceananigans.Fields: @compute 
#using Oceanostics

function get_budget_outputs_tuple(model; )

    # Output: primitive fields + computations
    b = model.tracers.b
    B̄ = model.background_fields.tracers.b
    B = B̄ + b # total buoyancy field
    # Cg = model.tracers.Cg

    advection = model.advection

    diffusivities = model.diffusivity_fields

    # c_immersed_bc = model.tracers.Cg.boundary_conditions.immersed
    # b_immersed_bc = model.tracers.b.boundary_conditions.immersed

    velocities = model.velocities    

    ########## ADVECTIVE FLUX DIVERGENCE
    # udiv_c = KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid; computed_dependencies=(velocities, Cg), parameters = (advection,))
   #maybe this instead?
    # udiv_c = KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid; computed_dependencies=(advection, velocities, Cg))

    udiv_B = KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid, advection, velocities, B)

    ######### DIFFUSIVE TERMS
    # ∇κ∇Cg = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, model.grid, model.closure, diffusivities, val_tracer_index, Cg, model.clock, model_fields, model.buoyancy)
    # ∇κ∇Cg_im = KernelFunctionOperation{Center, Center, Center}(immersed_∇_dot_qᶜ, model.grid, Cg, c_immersed_bc, model.closure, diffusivities, val_tracer_index, model.clock, model_fields)
    ∇κ∇B = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, model.grid, model.closure, diffusivities, Val(:b), 
                                    B, model.clock, model.velocities, model.buoyancy)

    # ∇κ∇B_im = KernelFunctionOperation{Center, Center, Center}(immersed_∇_dot_qᶜ, model.grid, B, b_immersed_bc, model.closure, diffusivities, val_tracer_index, model.clock, model_fields)

    ## ∇c TERMS
    # dCdx = Field(@at (Center, Center, Center) ∂x(Cg))
    # dCdy = Field(@at (Center, Center, Center) ∂y(Cg))
    # dCdz = Field(@at (Center, Center, Center) ∂z(Cg))

    ## ∇b TERMS
    dBdz = Field(@at (Center, Center, Center) ∂z(B))
    dBdx = Field(@at (Center, Center, Center) ∂x(B))
    dBdy = Field(@at (Center, Center, Center) ∂y(B))

    # cb = Field(@at (Center, Center, Center) Cg * b)

    ∇B = (dBdx^2 + dBdy^2 + dBdz^2)^0.5

    # outputs = merge(model.tracers,(; ∇κ∇Cg=∇κ∇Cg, ∇κ∇B=∇κ∇B, udiv_c = udiv_c, udiv_B=udiv_B, ))
    outputs = (; udiv_B=udiv_B,  ∇κ∇B=∇κ∇B, ∇B=∇B, ∇κ∇B_field=∇κ∇B_field)

    return outputs

end
