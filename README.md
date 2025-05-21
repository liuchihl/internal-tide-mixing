# internal-tide-mixing

This branch works on the supercomputer Delta

Purpose of this branch:
CG solver could be implemented on A100 GPU with Oceananigans@v0.95.5, but has CUDA errors with either different diags (velocities), or dt reduced from 20 to 10, etc.. so I want to figure out which one is the culprit
The plan: 
1. Run this branch with latest Oceananigans (because we know the old version works fine), run to t=451TP. (this works)
2. if 1. works, then try run with H200 (other things are the same), run to t=452TP (this works)
3. if 2. works, save all diagnostics (blows up due to CFL with dt=20) 
4. run with dt=10, still encounters CFL issue
5. try dt=5, run a bit further but encounters: CUDA error: an illegal memory access was encountered
6. so dt=5 is too small. Let's try dt = 10 and print out the progress message, check:
   (1) how many iteration is needed to converge, if it always reaches maxiter, then maxiter is too small. (the result shows iteration always meets the max and residual increases in time, indicating the CG solver is not set correctly) 
7. set maxiter from 100 to 1000, see what difference it makes (monitor whether residual decreases, and check if iteration is smaller than 1000)