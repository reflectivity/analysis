# simulate helical magnetization in an iron film on Si
# to create simulated data along with the simulation
# from the command line:
# refl1d simulation.py --edit --simulate --noise=1
# where noise=1 corresponds to 1% errorbars

################################################################
# Code originally from Brian K.
# modified by Andrew Caruana to use FunctionalMagnetism
################################################################

# An example of how to use the FunctionalMagnetism() 
# function for a helix (Sinusoid Function) in Fe. 


from refl1d.names import *
from numpy import *
# LOAD DATA AND CREATE A PROBE
if 0:
    probe = load4('yourdata.refl')
# JUST DO A SIMULATION
if 1:
    # arguments in numpy.linspace are theta start, theta stop, and number of points to calculate
    # "L" is the wavelength in A
    xs_probes = [Probe(T=numpy.linspace(0.1, 2.5, 401), L=4.75) for xs in range(4)]
    probe = PolarizedNeutronProbe(xs_probes, Aguide=270.00, H=0.000)
# turn off spin-flip calculation
if 1:
    probe.xs[1],probe.xs[2] = None, None 
# initialize offsets
probe.pp.intensity.value = 1.0
probe.pp.sample_broadening.value = 0.05
probe.pp.theta_offset.value = 0.0
# offsets are spin-independent
probe.mm.intensity = probe.pp.intensity
probe.mm.theta_offset = probe.pp.theta_offset
probe.mm.sample_broadening = probe.pp.sample_broadening

if 0:
    probe.oversample(n=11)

# enter initial values for nuclear sld in 10^-6 A^-2
Fe= SLD(name="Fe",rho=8.024)
Si= SLD(name="Si",rho=2.074)

# EXTRA PARAMETERS
# Initial starting values for parameters
M1 = 1000     
lamb = 101 
phi = 0 
thM=270
# DEFINE A LAYER STRUCTURE
# define each layer as:
# material(thickness,roughness,Magnetism(rhoM,thetaM))
# thickness & roughness are in A
# rhoM is in 10^-6 A^-2
# thetaM is in degrees (270 is parallel to H)
# SAMPLE

def FuncMag(z,M1,lamb,phi,thM):
    '''
    z needs to be the first arg in the function definition.

    Then define the functional magnetism by:
    rhoM(z) = Function
    thetaM(z) = Function <-- This is optional, if left out
    it will default to thetaM = 270, which is along the 
    guide field.

    In this example, the M1 parameter is converted
    from mSLD (10^-6 A^-2) to units of (kA/m) by use of const C.

    In larger scripts it can be prudent to import this function
    from a separate module, to avoid issues with .pickle.
    '''
    C = 2.91043e-3
    rhoM = C*(M1 * np.sin(2*np.pi*z/lamb+phi))
    #thetaM = thM + z*0
    return rhoM # ,thetaM

sample=(Si(5000,0)
           |Fe(1000,0,magnetism=FunctionalMagnetism(FuncMag))  
           |air)

sample[1].magnetism.M1.value = M1
sample[1].magnetism.M1.name = "M1 (kA/m)"

sample[1].magnetism.lamb.value = lamb
sample[1].magnetism.lamb.name = "lambda (A)"
sample[1].magnetism.phi.value = phi
sample[1].magnetism.phi.name = "phi (rad)"

sample[1].magnetism.thM.value = thM

# CONSTRAINTS
# roughness is constant for each microslab and equal to that of the thick layer below	 
# for each condition, spacer rhoM profile defined in terms of M1, M2, lambda, and phi
# DEFINE FITTING PARAMETERS
# Beam parameters
if 1:
    probe.pp.intensity.range(0.01,3)
if 1:
    probe.pp.theta_offset.range(-0.03,0.03)
if 1:
    probe.pp.sample_broadening.range(0,0.2)
# Fe
if 1:
    sample[1].thickness.range(1, 2000)
if 1:
    sample[1].magnetism.M1.range(0,5000)
if 1:
    sample[1].magnetism.phi.range(0,2*np.pi)
if 1:
    sample[1].magnetism.lamb.range(0,1000) # for smaller lambda, decrease dz to avoid artifacts
if 1:
    Fe.rho.range(6,9)
if 1:
    sample[Si].interface.range(0,50)
    sample[1].interface.range(0,50)

# PROBLEM DEFINITION
# step_interfaces = False corresponds to Nevot-Croce Approximation,
# True corresponds to direct calculation from the profile
# for the latter, microslabbing is defined by dz (in Angstroms)
M = Experiment(sample=sample, probe=probe, dz=2,step_interfaces=True)
problem = FitProblem(M)