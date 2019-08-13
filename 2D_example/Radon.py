import numpy as np
import m8r
import pylops
from pylops.utils.seismicevents import makeaxis
from pylops.optimization.sparsity import *


mat = m8r.Input('dat_full-151.rsf')
n1 = mat.int("n1")
n2 = mat.int("n2")
D =mat.read(shape=(n2,n1))

D = D.T
nr,nt = D.shape
N = nt*nr
print(nr,nt)
# model parameters

par = {'ox':0, 'dx':10, 'nx':nr,
       'ot':0, 'dt':0.0025, 'nt':nt}
taxis, t2, xaxis, y = makeaxis(par)
nx_mod = 1601
nz_mod = 801
dx_mod = 2.5
dx = 10
dz_mod = 2.5
nt_mod = 6001
dt_mod = 0.0005
dt = 0.0025

interval = 6
node = np.arange(0,nr,interval) 
jitter = np.random.randint(interval,size=node.size-1)
jitter_last = np.random.randint(nr - node[-1])
jitter = np.concatenate((jitter,jitter_last),axis=None)
idr = node + jitter
cod = np.arange(0,nt*nr).reshape(nr,nt)
idx = cod[idr,:].flatten()
R = pylops.Restriction(N, idx)
D_dec = R * D.flatten()
D_adj = (R.H * D_dec).reshape(nr,nt)

# mask for direct waves
vel_ocean = 1490

x1 = np.arange(150)
x2 = np.arange(150,301)
t1 = (150-x1)*dx/(vel_ocean*dt)
t2 = (x2-150)*dx/(vel_ocean*dt) 
t_vec = np.concatenate((t1,t2),axis=None) + 140

# mask for sediment reflection
t3 = np.sqrt(830**2 + ((150-x1)*dx)**2/((vel_ocean*dt)**2))
t4 = np.sqrt(830**2 + ((x2-150)*dx)**2/((vel_ocean*dt)**2))
t_vec2 = np.concatenate((t3,t4),axis=None)

mask_t2 = np.zeros((nt,nr))
for i in np.arange(nr):
    mask_t2[int(t_vec[i]):int(t_vec2[i]),i] = 1

Sop_t = pylops.Smoothing2D(nsmooth=[11,3], dims=[nt, nr])

mask_ts = (Sop_t*mask_t2.flatten()).reshape(nt,nr)

mask_2 = pylops.Diagonal(mask_ts.T)
D_adj_mask = (mask_2*D_adj.flatten()).reshape(nr,nt)
D_dec_mask = (R*D_adj_mask.flatten()).reshape(idr.size,nt)

D_mask2 = mask_2*D.flatten()
D_aft = D_mask2.reshape(nr,nt)


scail = abs((D_aft).max())
ND = (D_aft)/ scail

nwins = 10
winsize = 40
overlap = (nwins*winsize-nr)/(nwins-1)
pmax = 1e-4
npx = 100
px = np.linspace(-pmax,pmax, npx)

dimsd = D.shape
dims = (nwins*npx, par['nt'])

Op = \
    pylops.signalprocessing.Radon2D(taxis, np.linspace(-par['dx']*winsize//2,
                                                   par['dx']*winsize//2,
                                                   winsize),
                                    px, centeredh=False, kind='hyperbolic',
                                    engine='numba',dtype='complex128')

Slid = pylops.signalprocessing.Sliding2D(Op, dims, dimsd,
                                         winsize, overlap,
                                         tapertype=None)

# radon = Slid.H * ND.flatten()
# radon = radon.reshape(dims)

# derivatives

# calculate derivatives from original data(with masks and highpass)
F = pylops.signalprocessing.FFT2D(dims=(nr,nt),nffts=(nr,nt))
fre_sqz = F*ND.flatten() # change it if scailing is used

dx=10
kn=1/(2*dx)
ks = np.fft.fftfreq(nr, d=dx)

dt=0.0025
fn=1/(2*dt);

coeff1 = 1j*2*np.pi*ks
coeff2 = -(2*np.pi*ks)**2

coeff1_m = np.tile(coeff1,nt)
coeff2_m = np.tile(coeff2,nt)

coeff1 = np.tile(coeff1[:,np.newaxis], [1,nt])
coeff2 = np.tile(coeff2[:,np.newaxis], [1,nt])

D1op_hand = pylops.Diagonal(coeff1)
D2op_hand = pylops.Diagonal(coeff2)

D1_hand_fre = D1op_hand*fre_sqz
D2_hand_fre = D2op_hand*fre_sqz

D1_hand = F.H*D1_hand_fre
D2_hand = F.H*D2_hand_fre

# solve the linear equations
D2_dec = np.real(R*(D2_hand))
D1_dec = np.real(R*(D1_hand))
D_dec = R*ND.flatten()

Forward2 = pylops.VStack([R*mask_2*Slid, 
                          R*mask_2*F.H*D1op_hand*F*Slid, 
                          R*mask_2*F.H*D2op_hand*F*Slid
                          ])
rhs2 = np.concatenate((D_dec, D1_dec, D2_dec), axis=0)

####################
### LSQR solver ####
####################
xinv_ = \
    pylops.optimization.leastsquares.RegularizedInversion(Forward2, [], rhs2, 
                                                          **dict(damp=0, iter_lim=400, show=0))
# xista, niteri, costi = \
#     pylops.optimization.sparsity.FISTA(Forward2, rhs2, niter=200, eps=1e-4,
#                                       tol=1e-5, returninfo=True)
####################
### SPGL1 solver ###
####################
# xinv, pspgl, info = SPGL1(Forward, rhs,returninfo=True, 
#                              **dict(iterations=50))

xinv = scail*mask_2*Slid*xinv_
# xinv2 = scail*Slid*xista

xinv = np.real(xinv.reshape(nr,nt))
# xinv2 = np.real(xinv2.reshape(nr,nt))

xinv_fre = np.fft.fftshift(np.fft.fft2(xinv))
# xinv2_fre = np.fft.fftshift(np.fft.fft2(xinv2))

np.save('xinv_radon', xinv)