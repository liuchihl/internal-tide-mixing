# internal-tide-mixing

This branch goes back to this commit: https://github.com/liuchihl/internal-tide-mixing/commit/c5940477a9fe3bbc015ff8032b18c3b84a31ab5a
That usees maxiter=100, rtol=1e-9, H200. This has been shown that this branch works fine at least from t=451-452TP.

This new branch starts from there.

Changes:
1. tp_end=451; save x-z slices; tol = 1e-8; maxiter=500; Δt = 15
2. tp_end=452; save same diags as 1.; tol = 1e-8; maxiter=500; Δt = 15
3. tp_end=453; save all diags, including particles and Bbudgets; tol = 1e-8; maxiter=500; Δt = 15 (failed at t=452.04166667096825 with CUDA error)
4. try tp_end=453; try analysis round 2 (without saving everything, no particles and Bbudgets); tol = 1e-8; maxiter=500; Δt = 15, increase progress message frequency (get NaNs at 452.126404034896)
5. try tp_end=453 again, and try analysis round 2, increase progress message period to Δtᵒ÷60; cleanup GPU/CPU memory usage (no preconditioner would cause a quick increase of CG residual)
5. try tp_end=453 again, and try analysis round 2; but this time try maxiter=2500 and tol=1e-8, the goal is to test how many iteration is needed for the solution to converge. (get NaNs)
5. try tp_end=453 again, and try analysis round 2; use maxiter=2500 and tol=1e-8, use tidal period=44880 and dt=10, so the saving periods (snapshot, avg) are all multiple of dt (still receiving NaNs)
6. the problem for 5. is that the callback period of progress message should also be a multiple integer of dt. So try callback period: Δtᵒ/55=44880/24/55=34; maxiter=1000 (becaue 2500 gets impractically slow); tidal period remains the same = 4480. (runs through 452.234TP without crashing, which is the furthest I've ever get!, but I have to terminate it because I won't get the checkpoint at 453 since the simulation requires ~86hours to finish 1TP of run. So let's try maxiter=500, which saves more time.)
7. run maxiter=500 and save all diags; and set all saving interval/progress message interval to be an integer multiple of dt. the rest is the same as above. (CUDA error)
8. run maxiter=5000 and save the simple diags. The goal is to use high maxiter initially, at later timesteps, iteration could be less to converge. (iteration is always 5000, which doesn't help converging fast).
9. Split diags into "simple" and "complex", because saving at once would cause CUDA error. (simple causes CUDA error)
10. run "simple" diags, but only save uhat (this is just a test, CUDA error appears)
11. try getting rid of particles (NaN still appears)
12. don't save any outputs, just leave the progress message period = timestep (fails with NaN)
13. try progress message interval = 30 (NaN immediately)
14. try progress message interval = 34 (since it worked in 6.)
15. try progress message interval = 10, but set align_time_step=false in the Simulation. I want to make sure that timesteps wouldn't change due to the scheduler.

