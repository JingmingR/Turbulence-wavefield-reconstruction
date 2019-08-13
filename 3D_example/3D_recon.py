import m8r
import numpy as np
import pylops

"""
This is a demonstration of 3D reconstruction
with multichannel sampling theorem!!
"""
def loadrsf(file_name):
    mat = m8r.Input('/quanta1/home/ruan/OceanTurb_share_Jan2019/%s.rsf' % file_name)
    n1 = mat.int("n1")
    n2 = mat.int("n2")
    D = mat.read(shape=(n2,n1))
    shot = int(file_name[9:])
    return D, shot

D_all = np.zeros((1081,301,301))

nt = 1081
ns = 301
nr = 301

for i in range(0,ns):
    D,shot = loadrsf('dat_full-%s' % i)
    D_all[:,:,i] = D


#D = np.load('/quanta1/home/ruan/Script/D_cut.npy')
D = D_all[:,10:291,10:291]
nt,nr,ns = D.shape
print(nt,nr,ns)

dt = 0.0025
ds = 10
dr = 10

# maksing on space-time domain
dt = 0.0025
dx = 10
vel = 1490
shot = np.arange(0,nr)
mask_t = np.zeros((nt,nr,ns))

for i in range(0,ns):
    loc = shot[i]
    x1 = np.arange(loc)
    x2 = np.arange(loc,nr)
    t1 = (loc-x1)*dx/(vel*dt)
    t2 = (x2-loc)*dx/(vel*dt) 
    t_vec = np.concatenate((t1,t2),axis=None) + 140

    # mask for sediment reflection
    t3 = np.sqrt(830**2 + ((loc-x1)*dx)**2/((vel*dt)**2))
    t4 = np.sqrt(830**2 + ((x2-loc)*dx)**2/((vel*dt)**2))
    t_vec2 = np.concatenate((t3,t4),axis=None)

    for k in np.arange(nr):
        mask_t[int(t_vec[k]):int(t_vec2[k]),k,i] = 1
    
    Sop_t = pylops.Smoothing2D(nsmooth=[11,3], dims=[nt, nr])
    mask_t[:,:,i] = (Sop_t*mask_t[:,:,i].flatten()).reshape(nt,nr)
    
mask_tt = pylops.Diagonal(mask_t.flatten())
D = (mask_tt*D.flatten()).reshape(nt,nr,ns)

# mask on Fourier domain
kn=1/(2*dx)
dk=2*kn/nr

mask_fre_loc = np.zeros((nt,nr,ns))
vel = 1450

vn = 1/(2*dt)
dv = 2*vn/nt
vs = np.arange(-vn,vn,dv)

k1 = np.zeros(nt)
k2 = np.zeros(nt)
for i in np.arange(350,540):
    k1[i] = ((-vn + i*dv)/(vel)) /dk +150
    k2[i] = ((-vn + i*dv)/(-1*vel)) /dk +150
#     mask_fre_loc[i,:,int(k1[i]):int(k2[i])] = 1
#     mask_fre_loc[i,int(k1[i]):int(k2[i]),:] = 1
    mask_fre_loc[i,int(k1[i]):int(k2[i]),int(k1[i]):int(k2[i])] = 1
for i in np.arange(540,730):
    k1[i] = ((-vn + i*dv)/(vel) ) /dk +150
    k2[i] = ((-vn + i*dv)/(-1*vel) ) /dk +150
#     mask_fre_loc[i,:,int(k2[i]):int(k1[i])] = 1
#     mask_fre_loc[i,int(k2[i]):int(k1[i]),:] = 1
    mask_fre_loc[i,int(k2[i]):int(k1[i]),int(k2[i]):int(k1[i])] = 1
    
mask_fre_loc[300:350,:,:]=1
mask_fre_loc[730:780,:,:]=1
mask_fre_loc = np.fft.ifftshift(mask_fre_loc)
mask_fre = pylops.Diagonal(mask_fre_loc)
# normalize
scail = abs((D).max())
ND = (D)/ scail

# derivatives
dx=10
krn=1/(2*dx)
krs = np.fft.fftfreq(nr, d=dx)

ds =10
ksn = 1/(2*ds)
kss = np.fft.fftfreq(ns, d=ds)

mat_first_r = 1j*2*np.pi*krs
mat_first_s = 1j*2*np.pi*kss
mat_second_r = -(2*np.pi*krs)**2
mat_second_s = -(2*np.pi*kss)**2

fir_deriv_r = np.tile(mat_first_r[np.newaxis, :,np.newaxis ], [nt, 1, ns])
fir_deriv_s = np.tile(mat_first_s[np.newaxis, np.newaxis, : ], [nt, nr, 1])
sec_deriv_r = np.tile(mat_second_r[np.newaxis, :,np.newaxis ], [nt, 1, ns])
sec_deriv_s = np.tile(mat_second_s[np.newaxis, np.newaxis, : ], [nt, nr, 1])

D1op_r = pylops.Diagonal(fir_deriv_r)
D2op_r = pylops.Diagonal(sec_deriv_r)
D1op_s = pylops.Diagonal(fir_deriv_s)
D2op_s = pylops.Diagonal(sec_deriv_s)

F = pylops.signalprocessing.FFTND(dims=(nt, nr, ns),nffts=(nt, nr, ns))
fre = F*D.flatten()
fre_shift = np.fft.fftshift(fre.reshape(nt,nr,ns))
fre_ND = F*ND.flatten()

D1_r = (F.H*D1op_r*fre_ND)
D2_r = (F.H*D2op_r*fre_ND)
D1_s = (F.H*D1op_s*fre_ND)
D2_s = (F.H*D2op_s*fre_ND)
D1_rs = (F.H*D1op_r*D1op_s*fre_ND)
D1sD2r = (F.H*D1op_s*D2op_r*fre_ND)
D2sD1r = (F.H*D2op_s*D1op_r*fre_ND)
D2sD2r = (F.H*D2op_s*D2op_r*fre_ND)

# sampling operator
#interval = 4
#node = np.arange(0,nr,interval) 
#jitter = np.random.randint(interval,size=node.size-1)
#jitter_last = np.random.randint(nr - node[-1])
#jitter = np.concatenate((jitter,jitter_last),axis=None)
#idr = node + jitter
idr = np.load('idr_14.npy')

N = nt*nr*ns
cod = np.arange(0,N).reshape(nt,nr,ns)
#idr = np.arange(0,nr,3)
#ids = np.arange(0,ns,3)

idrn = cod[:,idr,:]
idx = idrn[:,:,idr].flatten()

R = pylops.Restriction(N, idx)

D_dec = R*ND.flatten()
D_adj = (R.H*D_dec).reshape(nt,nr,ns)
D1_r_dec = np.real(R*(D1_r.flatten()))
D1_s_dec = np.real(R*(D1_s.flatten()))
D2_r_dec = np.real(R*(D2_r.flatten()))
D2_s_dec = np.real(R*(D2_s.flatten()))
D1_rs_dec = np.real(R*(D1_rs.flatten()))
D1sD2r_dec = np.real(R*(D1sD2r.flatten()))
D2sD1r_dec = np.real(R*(D2sD1r.flatten()))
D2sD2r_dec = np.real(R*(D2sD2r.flatten()))


Forward = pylops.VStack([R*mask_tt*F.H*mask_fre, 
                         R*mask_tt*F.H*D1op_r*mask_fre, 
                         R*mask_tt*F.H*D1op_s*mask_fre,
                         R*mask_tt*F.H*D1op_r*D1op_s*mask_fre,
                         R*mask_tt*F.H*D2op_r*mask_fre,
                         R*mask_tt*F.H*D2op_s*mask_fre,
                         R*mask_tt*F.H*D1op_s*D2op_r*mask_fre,
                         R*mask_tt*F.H*D2op_s*D1op_r*mask_fre,
                         R*mask_tt*F.H*D2op_s*D2op_r*mask_fre])

rhs = np.concatenate((D_dec, 
                      D1_r_dec, 
                      D1_s_dec,
                      D1_rs_dec,
                      D2_r_dec,
                      D2_s_dec,
                      D1sD2r_dec,
                      D2sD1r_dec,
                      D2sD2r_dec), axis=0)

it=1000
xinv_ = \
    pylops.optimization.leastsquares.RegularizedInversion(Forward, [], rhs, 
                                                          **dict(damp=0, iter_lim=it, show=0))

xinv = scail*mask_tt*F.H*mask_fre*xinv_
xinv = np.real(xinv.reshape(nt,nr,ns))
xinv_fre = np.fft.fftshift(np.fft.fftn(xinv))

relative = np.abs((xinv - D) / np.amax(D))
relative_fre = np.abs((xinv_fre - fre_shift) / np.amax(np.abs(fre_shift)))

np.savez_compressed('xinv_ir14_it1000_maskex', xinv=xinv, relative=relative,relative_fre=relative_fre)
