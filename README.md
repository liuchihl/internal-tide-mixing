# internal-tide-mixing

This branch works on the supercomputer Delta

Purpose of this branch:
CG solver could be implemented on A100 GPU with Oceananigans@v0.95.5, but has CUDA errors with either different diags (velocities), H200, or dt reduced from 20 to 10, etc.. so I want to figure out which one is the culprit
The plan: 
1. Run this branch with latest Oceananigans using A100 (because we know the old version works fine), run to t=451TP with dt=20, maxiter=100, retol=1e-9. (this works)
2. if 1. works, then try run with H200 (other things are the same), run to t=452TP with dt=20, maxiter=100, retol=1e-9 (this works)
3. if 2. works, save all diagnostics (blows up due to CFL with dt=20, maxiter=100, retol=1e-9) 
4. run with dt=10, still encounters CFL issue
5. try dt=5, run a bit further but encounters: CUDA error: an illegal memory access was encountered
6. so dt=5 might be too small. So try dt = 10 and print out the progress message, check:
   (1) how many iteration is needed to converge, if it always reaches maxiter, then maxiter is too small. (the result shows iteration always meets the max and residual increases in time, indicating maxiter is too small) 
7. change maxiter from 100 to 1000, see what difference it makes (monitor whether residual decreases, and check if iteration is smaller than 1000)
    The result shows that iteration still always = maxiter but the residual has dropped to 1e-8 which is great! Only problem is that this is too slow, i.e., 1TP needs about 66hours wall clock time. So tolerance should be less strict and maxiter should be less in order to make progress.
8. next step is to test other types of preconditioners (SparseInverse) (failed when using it on CUDA)
    (1) see how many iteration needed for each CG solve, and the residual; 
    (2) plot the x-z slice to see the result
9. Find the best way, rerun the spinup period from t=450 to 452TP because maxiter=100 is too small.