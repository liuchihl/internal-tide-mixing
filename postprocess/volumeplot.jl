using PlotlyJS, ImageFiltering, Blink
using NCDatasets
filename_field = "output/higher_top_resolution/internal_tide_theta=0.0036_realtopo3D_Nx=500_Nz=250_10-30_threeD_timeavg.nc"
ds_field = Dataset(filename_field,"r")

# grids
zC = ds_field["zC"]; Nz=length(zC[:])
zF = ds_field["zF"]; zF = zF[:];
xC = ds_field["xC"]; Nx=length(xC[:])
xF = ds_field["xF"]; 
yC = ds_field["yC"]; Ny=length(yC[:])
yF = ds_field["yF"]; 
t = ds_field["time"][:];

n=58
b = ds_field["b"][:,:,:,n:n];
Bz = ds_field["Bz"][:,:,:,n:n];
epsilon = ds_field["ε"][:,:,:,n:n];
uhat = ds_field["uhat"][:,:,:,n:n];    # true u
what = ds_field["what"][:,:,:,n:n];    # true w
chi = ds_field["χ"][:,:,:,n:n];   
what_cen = (what[:,:,1:end-1,1] .+ what[:,:,2:end,1])./2 # what at center

# piecewise linear interpolation of what_cen from [center,center,center] to [face,center,center]
wtemp = (vcat(what_cen[end:end,:,:],what_cen[1:end-1,:,:]) .+ what_cen[:,:,:])./2
θ = 0.0036
u = uhat[:,:,:,1]*cos(θ) .+ wtemp*sin(θ)

# mask out
Bz[b.==0] .= NaN
epsilon[epsilon.==0] .= NaN;
chi[b.==0] .= NaN;
u[uhat[:,:,:,1].==0] .= NaN;
uhat[uhat.==0] .= NaN;
b[b.==0] .= NaN;
what[what.==0] .= NaN;
int=12
value = u[1:int:end,1:int:end,1:int:150];
X, Y, Z = mgrid(xC[1:int:end], yC[1:int:end], zC[1:int:150])

trace = PlotlyJS.volume(
    x=X[:], y=Y[:], z=Z[:],
    value=value[:],
    isomin=-0.02,
    isomax=0.02,
    opacity=0.6,
    surface_count=50,
    colorscale=reverse(colors.RdBu),
)
layout = PlotlyJS.Layout(scene_xaxis_showticklabels=true,
                  scene_yaxis_showticklabels=true,
                  scene_zaxis_showticklabels=true,
                  scene_aspectratio_x=1,
                  scene_aspectratio_y=2,
                  scene_aspectratio_z=1.5)

pt=PlotlyJS.plot(trace,layout);
save("output/higher_top_resolution/volume_u.png",pt)