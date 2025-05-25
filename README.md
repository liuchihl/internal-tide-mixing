# internal-tide-mixing

Purpose of this branch:
CG solver could be implemented on A100 GPU with Oceananigans@v0.95.5, but has CUDA errors with either different diags (velocities), H200, or dt reduced from 20 to 10, etc.. so I want to figure out which one is the culprit
The plan: 
1. Run this branch with latest Oceananigans using A100 (because we know the old version works fine), run to t=451TP with dt=20, maxiter=100, retol=1e-9. (this works)
2. if 1. works, then try run with H200 (other things are the same), run to t=452TP with dt=20, maxiter=100, retol=1e-9 (this works)
3. if 2. works, save all diagnostics (blows up due to CFL with dt=20, maxiter=100, retol=1e-9) 
4. run with dt=10, still encounters CFL issue
5. try dt=5, run a bit further but encounters (save all diags): CUDA error: an illegal memory access was encountered
6. so dt=5 might be too small. So try dt = 10 and print out the progress message, check:
   (1) how many iteration is needed to converge, if it always reaches maxiter, then maxiter is too small. (the result shows iteration always meets the max and residual increases in time, indicating maxiter is too small) 
7. change maxiter from 100 to 1000 (only 2 diags; no 3D save), see what difference it makes (monitor whether residual decreases, and check if iteration is smaller than 1000)
    The result shows that iteration still always = maxiter but the residual has dropped to 1e-8 which is great! Only problem is that this is too slow, i.e., 1TP needs about 66hours wall clock time. So tolerance should be less strict and maxiter should be less in order to make progress.
8. next step is to test other types of preconditioners if they give faster convergence (SparseInverse and AsymptoticInverse) (these preconditioners are not implemented yet for immersed boundary on GPU, only CPU would work). So use previous preconditioner AsymptoticPoissonPreconditioner().
9. The test using maxiter=500, tol=1e-8, dt=15: 1TP takes 25 hours of run with only 2 diagnostics (xy-slice) using H200 (this is too slow, but the residual stays around 1e-8 to 1e-7 without increasing, which is great)
10. try maxiter=300 and slightly higher tolerance (5e-8) and dt=15 (I tried this and save all diags, I met the CUDA error) 
11. increasing maxiter=500, tol=5e-8, and save all diags meet CUDA error (the residual increases to 70 and CUDA error appears)
    From 9-11, seems like saving certain diags might cause CUDA error... 
12. since 9. works, use that configuration but save more diags and see what happens (CUDA error appears..)
    try FFT and we might see the same CUDA error because the memory is not enough
13. from 9 and 12, it is sure that some diags introduce the CUDA error. So figuring out which diagnostics is the culprit is important.
    (1) all diags but without particles: CUDA error: an illegal memory access was encountered; CG residual: 1.67e+02 (particles are not the main culprit)
        The residual increases greatly after saving the 3D data (not sure if saving 3D data is related) (results shows 3D data is not related)
    (2) same as (1) but without the 3D saves (I suspect 3D save is the culprit, because in 7., there was no 3D saves) (this still failed, try get rid of Oceanostics)
    (3) same as (2) but with FFT (no issue is found, so the issue is indeed from CG)
    (4) same as (2) but without Oceanostics (same CUDA error, same failing time)





Conclusion:
1. CUDA error: an illegal memory access was encountered: this occurs when saving too many diags
2. residual increasing: if maxiter is too small
What's next?
1. Take out the particles 