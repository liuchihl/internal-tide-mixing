# internal-tide-mixing

This branch works on the supercomputer Delta

Purpose of this branch:
CG solver could be implemented on A100 GPU with Oceananigans@v0.95.5, but cannot be run in either later version, different diags (velocities), or dt reduced from 20 to 10, etc..
The plan: 
1. Run this branch with latest Oceananigans (because we know the old version works fine)
2. if 1. works, then try run dt=10
3. if 2. works, then try different diags, including velocities
4. if 3. also works miraculously, switch this to H200 and see what happens

https://github.com/user-attachments/assets/658eb080-3dfd-4244-bf62-ee2b36022d22

