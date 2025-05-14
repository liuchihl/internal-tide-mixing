# internal-tide-mixing

This branch works on the supercomputer Delta

Purpose of this branch:
CG solver could be implemented on A100 GPU with Oceananigans@v0.95.5, but cannot be run in either later version, different diags (velocities), or dt reduced from 20 to 10, etc.. so I want to figure out which one is the culprit
The plan: 
1. Run this branch with latest Oceananigans (because we know the old version works fine), run to t=451TP. (this works)
2. if 1. works, then try run with H200 (other things are the same), run to t=452TP (this works)
3. if 2. works, save all diagnostics
4. if 3. works, run dt=10, if not, increase dt to 15 perhaps


