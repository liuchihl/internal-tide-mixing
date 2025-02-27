using CUDA: has_cuda_gpu
using Oceananigans
using Oceananigans.AbstractOperations: @at, ∂x, ∂y, ∂z
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Advection: div_Uc
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ
using Oceananigans.TurbulenceClosures: immersed_∇_dot_qᶜ
using Oceanostics: validate_location

#import Oceananigans.TurbulenceClosures: viscosity, diffusivity
using Oceananigans.Fields: @compute 


function get_budget_outputs_tuple(model; )

    # Output: primitive fields + computations
    b = model.tracers.b
    B̄ = model.background_fields.tracers.b
    B = B̄ + b # total buoyancy field
    advection = model.advection

    diffusivities = model.diffusivity_fields

    b_immersed_bc = model.tracers.b.boundary_conditions.immersed
    velocities = model.velocities    
    # ########## buoyancy tendency
    # @inline ∂ₜcᶜᶜᶜ(i, j, k, grid, val_tracer_index::Val{tracer_index},
    #                            val_tracer_name,
    #                            advection,
    #                            closure,
    #                            c_immersed_bc,
    #                            buoyancy,
    #                            biogeochemistry,
    #                            background_fields,
    #                            velocities,
    #                            tracers, args...) where tracer_index =
    # @inbounds tracer_tendency(i, j, k, grid,
    #                             val_tracer_index,
    #                             val_tracer_name,
    #                             advection,
    #                             closure,
    #                             c_immersed_bc,
    #                             buoyancy,
    #                             biogeochemistry,
    #                             background_fields,
    #                             velocities,
    #                             tracers,
    #                             args...)

    # function TracerTendency(model::NonhydrostaticModel, tracer_name; location = (Center, Center, Center))
    #     validate_location(location, "TracerTendency")
    #     tracer_index = findfirst(n -> n === tracer_name, propertynames(model.tracers))
    #     dependencies = (Val(tracer_index),
    #                     Val(tracer_name),
    #                     model.advection,
    #                     model.closure,
    #                     model.tracers[tracer_name].boundary_conditions.immersed,
    #                     model.buoyancy,
    #                     model.background_fields,
    #                     model.velocities,
    #                     model.tracers,
    #                     model.auxiliary_fields,
    #                     model.diffusivity_fields,
    #                     model.forcing[tracer_name],
    #                     model.clock)
    
    #     return KernelFunctionOperation{Center, Center, Center}(∂ₜcᶜᶜᶜ, model.grid, dependencies...)
    # end

    # dbdt = TracerTendency(model, :b)


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
    # ∇⋅qc = -∇⋅(κ∇B)
    ∇κ∇B = ∇κ∇B_interior + ∇κ∇B_im
    
    outputs = (; div_uB=div_uB, ∇κ∇B= -∇κ∇B, )

    return outputs

end
