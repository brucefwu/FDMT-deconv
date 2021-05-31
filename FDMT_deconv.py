## Written by Bruce Wu

# import packages
import numpy as np
import matplotlib.pyplot as plt
from FDMT_functions import FDMT

# define functions
def t_pulse(t_2, f_2, f, DM):
    """
    Function for generating the time values of a pulse given the frequency
    values. Eq. 1 in FDMT paper.
    t_pulse, t_2 in s
    f_2, f in MHz
    DM in pc cm^-3
    """
    return t_2 + 4.148808*DM*((f/1000.)**(-2) - (f_2/1000.)**(-2))/1000.

def mod_FDMT(im):
    """
    Modular FDMT. Dispersion curves which leave the left boundary of the
    image come back through the right boundary. Achieved by duplicating
    the last N_f time bins of the image to the front, taking the FDMT, then
    cutting the duplicated time bins off. This makes the FDMT output in the
    region where the y index is greater than the x index more reliable.
    """
    # make modular image
    mod_im = np.empty((N_f, N_f+N_t)) # time axis extended by N_f
    mod_im[:,:N_f] = im[:,N_t-N_f:] # duplicate last N_f times in front
    mod_im[:,N_f:] = im # the rest is original image
    # take FDMT and truncate
    A = FDMT(mod_im, f_min-df/2., f_max+df/2., N_f, 'float64')
    return np.delete(A, np.arange(N_f), axis=1)

def t_scat(t_G, f):
    """
    Scattering timescale in seconds at f in MHz. t_G is the timescale at 
    1GHz, in seconds.
    """
    return t_G*((f/1000.)**(-4))

def S(F, tau, t):
    """
    FRB scattering profile at one frequency. The flux density S is in Jy 
    when the fluence F is in Jy*s and the scattering timescale tau is in s.
    t is also in s and t=0 corresponds to the first arrival of the signal.
    """
    return (F/tau)*np.exp(-t/tau)

def S_bin(F, t1, t2, tau):
    """
    Flux density in Jy which falls into a bin bounded by t1 and t2, in s.
    F is the fluence in Jy*s and tau is the scattering timescale in s.
    """
    return F*(np.exp(-t1/tau) - np.exp(-t2/tau))/(t2 - t1)






# load noise
sub_image = np.load('noise_2.npy')
nf, N_t = np.shape(sub_image) # nf is not power of two

# data parameters
N_f = 128 # padded up to nearest power of two
f_min = 169.755 # MHz
f_max = 210.395
dt = 0.5 # s
df = 0.32 # MHz
t = np.arange(N_t)*dt
f = np.arange(N_f)*df + f_min
const = 4.148808*((f_min/1000.)**(-2) - (f_max/1000.)**(-2)) # for converting 
                                                        # delay to DM
DMs = np.arange(N_f)*dt*1000./const # maximum delay is maximum bins
                             # spanned by pulse as inputed into FDMT
                             # multiplied by the timestep, then turned into
                             # a DM
dDM = DMs[1] - DMs[0] # DM step

# generate pulse
DM = 839.22 # dispersion measure of pulse, in pc cm^-3
t_2 = 170.4 # where pulse exits image at lowest frequency, in seconds
t_p = t_pulse(t_2, f_min, f[:nf], DM) # find pulse times
t_G = 0.0067 # s
Flu = 20. # Jy*s

# introduce pulse into image
bins = np.concatenate((t, np.array([t[-1]+dt]))) # add final bin boundary
for i in xrange(nf):
    k = np.digitize(np.array([t_p[i]]), bins)[0] # find in which bin the 
    # signal arrives
    sub_image[i,k-1] += S_bin(Flu, 0, bins[k]-t_p[i], t_scat(t_G, f[i]))
    # add signal into first bin
    for j in xrange(k, N_t):
	sub_image[i,j] += S_bin(Flu, bins[j]-t_p[i], bins[j+1]-t_p[i], \
	                        t_scat(t_G, f[i]))
	# add signal to subsequent bins

# pad up to power of two freqs
image = np.zeros((N_f, N_t))
image[:nf,:] = sub_image # fill in nf < N_f bins with data
sub_image = 0

# perform FDMT
A = mod_FDMT(image) # take modular FDMT
DM_max, t_max = np.unravel_index(np.argmax(A), np.shape(A))
print DMs[DM_max]


# Response function
G_image = np.zeros((N_f, N_t), dtype='float64')
t_G = t_pulse(t[N_t/2], f_min, f, DMs[DM_max])
y = range(N_f) # indices of frequency array
x = np.digitize(t_G, t).tolist() # find pulse time indices
G_image[y,x] += 1. # add certain value to image at pulse positions
G_image[nf:,:] = 0.
G = mod_FDMT(G_image)
hann0 = np.hanning(N_f)
hann1 = np.hanning(N_t)
hann2 = np.sqrt(np.outer(hann0, hann1)) # 2d hann window
F_A = np.fft.fft2((A - A.mean())*hann2) # detrend and hann
F_G = np.fft.fft2((G - G.mean())*hann2)
#F_A = np.fft.rfft2(A)
#F_G = np.fft.rfft2(G)
#F_G[np.where(np.absolute(F_G) < 0.5)] = 1. + 0.j
deconv = np.real(np.fft.ifftshift(np.fft.ifft2(F_A/F_G)))
DM_deconv_max, t_deconv_max = np.unravel_index(np.argmax(deconv), \
                                               np.shape(deconv))






# plot

plt.figure()
plt.imshow(image, origin='lower', cmap='Greys_r', interpolation='nearest', \
            extent=(t[0], t[-1]+dt, f_min-df/2., \
            f_max+df/2.), aspect='auto')
plt.xlim(0., t[-1]+dt)
plt.ylim(f_min-df/2., f_max+df/2.)
cbar = plt.colorbar()
cbar.set_label('Jy/beam', size=16)
plt.xlabel('Time (s)', size=16)
plt.ylabel('Frequency (MHz)', size=16)
plt.title('Waterfall', size=18)

plt.figure()
plt.imshow(G_image, origin='lower', cmap='Greys_r', interpolation='nearest', \
            extent=(t[0], t[-1]+dt, f_min-df/2., \
            f_max+df/2.), aspect='auto')
cbar = plt.colorbar()
plt.xlabel('Time (s)', size=16)
plt.ylabel('Frequency (MHz)', size=16)
plt.title('Waterfall for Response Function', size=18)

plt.figure()
plt.imshow(A, origin='lower', cmap='hot', interpolation='nearest', \
            extent=(t[0], t[-1]+dt, DMs[0]-dDM/2., DMs[-1]+dDM/2.), \
            aspect='auto')
plt.colorbar()
plt.xlabel('Time (s)', size=16)
plt.ylabel('Dispersion Measure (pc cm^-3)', size=16)
plt.title('Modular FDMT', size=18)

plt.figure()
plt.imshow(G, origin='lower', cmap='hot', interpolation='nearest', \
            extent=(t[0], t[-1]+dt, DMs[0]-dDM/2., DMs[-1]+dDM/2.), \
            aspect='auto')
plt.colorbar()
plt.xlabel('Time (s)', size=16)
plt.ylabel('Dispersion Measure (pc cm^-3)', size=16)
plt.title('Response Function', size=18)

plt.figure()
plt.imshow(deconv, origin='lower', cmap='hot', interpolation='nearest', \
            extent=(t[0], t[-1]+dt, DMs[0]-dDM/2., DMs[-1]+dDM/2.), \
            aspect='auto')
plt.colorbar()
plt.xlabel('Time (s)', size=16)
plt.ylabel('Dispersion Measure (pc cm^-3)', size=16)
plt.title('Deconvolved', size=18)

plt.figure()
plt.plot(t, deconv[DM_deconv_max,:])
plt.xlabel('Time (s)', size=16)
plt.title('Deconvolved Slice', size=18)

plt.show()