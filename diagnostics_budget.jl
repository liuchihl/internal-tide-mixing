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
    advection = model.advection

    diffusivities = model.diffusivity_fields

    b_immersed_bc = model.tracers.b.boundary_conditions.immersed
    velocities = model.velocities    

    ########## ADVECTIVE FLUX DIVERGENCE
    # ∇⋅(uB) = u⋅∇B
    div_uB = KernelFunctionOperation{Center, Center, Center}(div_Uc, model.grid, advection, velocities, B)

    ######### Divergence of buoyancy diffusive flux    
    # Diffusive flux = - ∇_dot_qᶜ - immersed_∇_dot_qᶜ
    
    # interior
    ∇κ∇B_interior = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, model.grid, model.closure, diffusivities, Val(:b), 
                                        B, model.clock, model.velocities, model.buoyancy) 
    # immersed boundaries
    ∇κ∇B_im = KernelFunctionOperation{Center, Center, Center}(immersed_∇_dot_qᶜ, model.grid, B, b_immersed_bc, 
                                        model.closure, diffusivities, Val(:b), model.clock, model.velocities)
    ∇κ∇B = ∇κ∇B_interior + ∇κ∇B_im
    
    # outputs = merge(model.tracers,(; ∇κ∇Cg=∇κ∇Cg, ∇κ∇B=∇κ∇B, udiv_c = udiv_c, udiv_B=udiv_B, ))
    # outputs = (; dxκdB=dxκdB,)
    outputs = (; div_uB=div_uB,  ∇κ∇B= -∇κ∇B)

    return outputs

end
