TITLE slo1 current for C. elegans neuron model (suppose to interact with unc2)


COMMENT
channel type: calcium-related potassium channel

slo1-unc2 1:1 stoichiometry

model and parameters from:
Nicoletti, M., et al. (2019). "Biophysical modeling of C. elegans neurons: Single ion currents and whole-cell dynamics of AWCon and RMD." PLoS One 14(7): e0218738.
ENDCOMMENT


UNITS {
    (nS) = (nanosiemens)
    (mV) = (millivolt)
    (pA) = (picoamp)
    (um) = (micron)
    (molar) = (1/liter)  : moles do not appear in units
    (uM) = (micromolar)
}


NEURON {
    SUFFIX slo1_unc2
    USEION k READ ek WRITE ik
    RANGE gbslo1, minf, tm, mcavinf, tmcav, hcavinf, thcav, m, hcav, mcav
}


PARAMETER {
    : slo1
    gbslo1 = 1 (nS/um2)
    wyx = 0.013 (1/mV)
    wxy = -0.028 (1/mV)
    wom = 3.15 (1/ms)
    wop = 0.16 (1/ms)
    kxy = 55.73 (uM/um3)
    nxy = 1.30
    kyx = 34.34 (uM/um3)
    nyx = 0.0001
    canci = 0.05 (uM/um2) : calcium concentration When the calcium channel is closed

    : unc2
    vhm = -12.2 (mV)
    ka = 4.0 (mV)
    vhh = -52.5 (mV)
    ki = 5.6 (mV)
    atm = 1.5 (ms)
    btm = -8.2 (mV)
    ctm = 9.1 (mV)
    dtm = 15.4 (mV)
    etm = 0.1 (ms)
    ath = 83.8 (ms)
    bth = 52.9 (mV)
    cth = -3.5 (mV)
    dth = 72.1 (ms)
    eth = 23.9 (mV)
    fth = -3.6 (mV)

    : cain
    FARADAY = 96485 (coul) : moles do not appear in units
	gsc = 0.04 (nS)   : single channel conductance, assumed equal to 40 pS for both L-type and P/Qtype calcium channels
    r = 0.013 (um)       : radius of the nanodomain
    dca = 250 (um2/s) : calcium diffusion coefficient
    kb = 500 (1/uM/s) : intracellular buffer rate constant
    btot = 30 (uM)    : total intracellular buffer concentration
}


ASSIGNED {
    v (mV)
    ek (mV)
    ik (pA/um2)
    minf
    tm (ms)
    mcavinf
    tmcav (ms)
    hcavinf
    thcav (ms)
}


STATE {
    m
    hcav
    mcav
}


BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbslo1 * m * hcav * (v + 80) : ek=-80
}


INITIAL {
    setparames(v)
    hcav = hcavinf
    mcav = mcavinf
    m = 0
}


DERIVATIVE states {
    setparames(v)
    mcav' = (mcavinf - mcav) / tmcav
    hcav' = (hcavinf - hcav) / thcav
    m' = (minf - m) / tm    
}


PROCEDURE setparames(v (mV)) {
    LOCAL alpha, beta, wm, wp, fm, fp, kom, kop, kcm, cain
    : unc2 model
    mcavinf = 1 / (1 + exp(-(v - vhm) / ka))
    hcavinf = 1 / (1 + exp((v - vhh) / ki))
    tmcav = atm / (exp(-(v - btm) / ctm) + exp((v - btm) / dtm)) + etm
    thcav = ath / (1 + exp(-(v - bth) / cth)) + dth / (1 + exp((v - eth) / fth))
    : cain
    cain = -gsc * (v - 60) * 10^9 / (8 * 3.1415926 * r * dca * FARADAY) * exp (-r / sqrt(dca / (kb * btot)))
    : slo1-unc2 model
    alpha = mcavinf / tmcav
    beta = 1 / tmcav - alpha
    wm = wom * exp(-wyx * v)
    wp = wop * exp(-wxy * v)
    fm = 1 / (1 + (cain / kyx) ^ nyx)
    fp = 1 / (1 + (kxy / cain) ^ nxy)
    kom = wm * fm
    kop = wp * fp
    kcm = wm * 1 / (1 + (canci / kyx) ^ nyx)
    minf = mcav * kop * (alpha + beta + kcm) / ((kop + kom) * (kcm + alpha) + beta * kcm)
    tm = (alpha + beta + kcm) / ((kop + kom) * (kcm + alpha) + beta * kcm)
}