COMMENT
Based on NMODL code for neuron_to_neuron_exc_syn

Code and comments have been normalized--that is,
they are now of the usual form employed in 
NMODL-specified synaptic mechanisms.

Oddities that save neither storage nor time, 
or preclude use with adaptive integration, 
or deliberately introduce errors in an attempt
to save a few CPU cycles, have been removed.

Unused variables and parameters have also been removed.

vpre is now a POINTER.

In the INITIAL block, the completely unnecessary repeat
execution of procedure rates() has been removed, 
and the state variable s is now set to the correct 
initial value.

The value of the current i is now calculated in the
BREAKPOINT block, instead of being calculated as a 
side-effect of calling rates().
ENDCOMMENT

TITLE excitatory to excitatory synapse

NEURON {
    POINT_PROCESS exc_syn_advance
    NONSPECIFIC_CURRENT i
    RANGE weight, conductance, i, vpre
    RANGE delta, k, Vth, erev
}

UNITS {
    (nA) = (nanoamp)
    (uA) = (microamp)
    (mA) = (milliamp)
    (A) = (amp)
    (mV) = (millivolt)
    (mS) = (millisiemens)
    (uS) = (microsiemens)
    (molar) = (1/liter)
    (kHz) = (kilohertz)
    (mM) = (millimolar)
    (um) = (micrometer)
    (umol) = (micromole)
    (S) = (siemens)
}

PARAMETER {
    weight = 1
    conductance = 4.9E-4 (uS)
    delta = 5 (mV)
    k = 0.5 (kHz)
    Vth = -20 (mV)
    erev = 30 (mV) : -10
}

ASSIGNED {
    v (mV)
    vpre (mV)
    inf
    tau (ms)
    i (nA)
}

STATE {
    s  
}

INITIAL {
    rates()
    s = inf : NTC original code did not properly initialize s
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = weight * conductance * s * (v - erev)
}

DERIVATIVE states {
    rates()
    s' = (inf - s)/tau
}

PROCEDURE rates() {
    inf = 1/( 1 + exp((Vth - vpre)/delta) )
    tau = (1 - inf)/k
}
