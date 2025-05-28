# internal-tide-mixing

This branch goes back to this commit: https://github.com/liuchihl/internal-tide-mixing/commit/c5940477a9fe3bbc015ff8032b18c3b84a31ab5a
That usees maxiter=100, rtol=1e-9, H200. This has been shown that this branch works fine at least from t=451-452TP.

This new branch starts from there.

Changes:
1. tp_end=451; save x-z slices; tol = 1e-8; maxiter=500; Δt = 15
2. tp_end=452; save same diags as 1.; tol = 1e-8; maxiter=500; Δt = 15
3. tp_end=453; save all diags, including particles and Bbudgets; tol = 1e-8; maxiter=500; Δt = 15 (failed at t=452.04166667096825 with CUDA error)
4. try tp_end=453; try analysis round 2 (without saving everything), including particles and Bbudgets; tol = 1e-8; maxiter=500; Δt = 15, increase progress message frequency
 
