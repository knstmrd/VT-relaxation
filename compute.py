import numpy as np
from math import exp, sinh, cosh, atan, factorial, gamma
from scipy import constants
from scipy import integrate
from netCDF4 import Dataset
import argparse

# all the VSS elastic collision data is here

vss_data = {
    'N2+N': {'dref': 4.088e-10, 'omega': 0.762},
    'N2+N2': {'dref': 4.04e-10, 'omega': 0.686},
    'N2+O': {'dref': 3.2220000000000004e-10, 'omega': 0.702},
    'N2+O2': {'dref': 3.604e-10, 'omega': 0.703},
    'N2+NO': {'dref': 4.391e-10, 'omega': 0.756},
    'N2+Ar': {'dref': 3.882e-10, 'omega': 0.703},

    'O2+N': {'dref': 3.7210000000000004e-10, 'omega': 0.757},
    'O2+N2': {'dref': 3.604e-10, 'omega': 0.703},
    'O2+O': {'dref': 3.734e-10, 'omega': 0.76},
    'O2+O2': {'dref': 3.8960000000000003e-10, 'omega': 0.7},
    'O2+NO': {'dref': 4.054e-10, 'omega': 0.718},
    'O2+Ar': {'dref': 3.972e-10, 'omega': 0.719},

    'NO+N': {'dref': 4.028e-10, 'omega': 0.788},
    'NO+N2': {'dref': 4.391e-10, 'omega': 0.756},
    'NO+O': {'dref': 3.693e-10, 'omega': 0.752},
    'NO+O2': {'dref': 4.054e-10, 'omega': 0.718},
    'NO+NO': {'dref': 4.2180000000000003e-10, 'omega': 0.737},
    'NO+Ar': {'dref': 4.049e-10, 'omega': 0.719},
}

# FHO_data = {'N2+N': {'beta': 4e10, 'E_Morse': 200 * constants.k, 'svt': 0.3183},
#             'N2+N2': {'beta': 3.8e10, 'E_Morse': 200 * constants.k, 'svt': 0.3183},
#             'N2+O': {'beta': 6e10, 'E_Morse': 200 * constants.k, 'svt': 0.3183},
#             'N2+O2': {'beta': 4.1e10, 'E_Morse': 150 * constants.k, 'svt': 0.3183},
#             'N2+NO': {'beta': 6e10, 'E_Morse': 107 * constants.k, 'svt': 0.3183},

#             'O2+N': {'beta': 6e10, 'E_Morse': 400 * constants.k, 'svt': 0.5},
#             'O2+N2': {'beta': 4.1e10, 'E_Morse': 150 * constants.k, 'svt': 0.3183},
#             'O2+O': {'beta': 7.5e10, 'E_Morse': 200 * constants.k, 'svt': 0.3183},
#             'O2+O2': {'beta': 4.5e10, 'E_Morse': 90 * constants.k, 'svt': 0.5},
#             'O2+NO': {'beta': 6e10, 'E_Morse': 113 * constants.k, 'svt': 0.3183},

#             'NO+N': {'beta': 6e10, 'E_Morse': 92 * constants.k, 'svt': 0.3183},
#             'NO+N2': {'beta': 6e10, 'E_Morse': 107 * constants.k, 'svt': 0.3183},
#             'NO+O': {'beta': 6e10, 'E_Morse': 97 * constants.k, 'svt': 0.3183},
#             'NO+O2': {'beta': 6e10, 'E_Morse': 113 * constants.k, 'svt': 0.3183},
#             'NO+NO': {'beta': 13e10, 'E_Morse': 119 * constants.k, 'svt': 0.3183},
#             }


FHO_data = {'N2+N': {'beta': 4.6e10, 'E_Morse': 1 * constants.k, 'svt': 0.99},
            'N2+N2': {'beta': 3.9e10, 'E_Morse': 1 * constants.k, 'svt': 0.9},
            'N2+O': {'beta': 7.3e10, 'E_Morse': 500 * constants.k, 'svt': 0.175},
            'N2+O2': {'beta': 3.9e10, 'E_Morse': 6 * constants.k, 'svt': 0.95},
            'N2+NO': {'beta': 4e10, 'E_Morse': 2 * constants.k, 'svt': 0.75},
            'N2+Ar': {'beta': 4e10, 'E_Morse': 4 * constants.k, 'svt': 0.8},

            'O2+N': {'beta': 7.3e10, 'E_Morse': 10 * constants.k, 'svt': 0.25},
            'O2+N2': {'beta': 4.1e10, 'E_Morse': 150 * constants.k, 'svt': 0.333333},
            'O2+O': {'beta': 2.6e10, 'E_Morse': 17000 * constants.k, 'svt': 0.2},
            'O2+O2': {'beta': 4.3e10, 'E_Morse': 40 * constants.k, 'svt': 0.99},
            'O2+NO': {'beta': 4.1e10, 'E_Morse': 150 * constants.k, 'svt': 0.333333},
            'O2+Ar': {'beta': 3.6e10, 'E_Morse': 200 * constants.k, 'svt': 0.99},

            'NO+N': {'beta': 5e10, 'E_Morse': 200 * constants.k, 'svt': 0.3183},
            'NO+N2': {'beta': 4.4e10, 'E_Morse': 20 * constants.k, 'svt': 0.9},
            'NO+O': {'beta': 7.9e10, 'E_Morse': 16000 * constants.k, 'svt': 0.06},
            'NO+O2': {'beta': 6.75e10, 'E_Morse': 1500 * constants.k, 'svt': 0.2},
            'NO+NO': {'beta': 6.25e10, 'E_Morse': 4500 * constants.k, 'svt': 0.03},
            'NO+Ar': {'beta': 5e10, 'E_Morse': 1100 * constants.k, 'svt': 0.1},
            }


# particle masses
masses = {'N2': 4.6517344343135997e-26,
          'O2': 5.3135256633152e-26,
          'NO': 4.9826300488143997e-26,
          'O': 2.6567628316576e-26,
          'N': 2.3258672171567998e-26,
          'Ar': 6.63385346650652e-26}

# molecule data: oscillator-reduced masses, characteristic vibrational and dissociation temperatures
mol_data = {'N2': {'osc_mass': 1.1629336085783999e-26, 'theta_v': 3393, 'theta_D': 113200},
            'O2': {'osc_mass': 1.3283814158288e-26, 'theta_v': 2273, 'theta_D': 59763},
            'NO': {'osc_mass': 1.240163831826812e-26, 'theta_v': 2740, 'theta_D': 75429}}

# for each molecule, mass(atom1)/mass(molecule), mass(atom2)/mass(molecule)
ram_masses = {'N2': [0.5, 0.5],
              'O2': [0.5, 0.5],
              'NO': [0.4668, 0.5332]}

red_masses = {}  # collision-reduced masses calculation
for p1 in masses:
    for p2 in masses:
        m1 = masses[p1]
        m2 = masses[p2]
        red_masses[p1 + '+' + p2] = m1 * m2 / (m1 + m2)


def fact_div_fact(start: int, end: int) -> float:
    """
    Helper function, returns end! / start!, where start and end are integers
    """
    return np.prod(np.arange(start + 1.0, end + 1.0, 1.0))


def Zv(Tv, vibr_spectrum):
    return np.sum(np.exp(-vibr_spectrum / (constants.k * Tv)))


def c_vibr(Tv, vibr_spectrum, mass):
    vae = np.sum(vibr_spectrum * np.exp(-vibr_spectrum / (constants.k * Tv))) / Zv(Tv, vibr_spectrum)
    vaesq = np.sum(vibr_spectrum**2 * np.exp(-vibr_spectrum / (constants.k * Tv))) / Zv(Tv, vibr_spectrum)
    return (vaesq - vae**2) / (constants.k * Tv * Tv * mass)


def cs_vss(g, dref, gref, omega):
    """
    VSS cross-section calculation (g is the velocity, dref and gref are reference parameters)
    """
    return constants.pi * dref**2 * (g / gref)**(1 - 2 * omega) / gamma(2.5 - omega)


def fact_div_fact(start: int, end: int) -> float:
    """
    Return the value start! / end!
    """
    return np.prod(np.arange(start + 1.0, end + 1.0, 1.0))


def svt(delta: int) -> float:
    """
    One of the suggested methods of calculating the steric factor, not used
    """
    s = delta
    if delta < 0:
        s *= -1.0
    return 1. / (constants.pi * s)


def vel_avg_vt(g: float, ve_before: float, ve_after: float, mass: float) -> float:
    """
    average the velocities before and after a collision
    """
    gn_sq = (ve_before - ve_after) * (2.0 / mass) + (g ** 2)
    if gn_sq < 0:
        return -1
    else:
        return 0.5 * (g + (gn_sq ** 0.5))


def vt_prob_g_only_fho_12(g: float, mass: float, beta: float, osc_mass: float,
                          ve_before: float, ve_after: float, i: int, delta: int,
                          ram, E_Morse, this_svt) -> float:
    """
    Compute the VT transition probability
    """
    res = 0.

    vel = vel_avg_vt(g, ve_before, ve_after, mass)

    if delta == 1:
        omega = (ve_after - ve_before) / constants.hbar
    elif delta == -1:
        omega = (ve_before - ve_after) / constants.hbar
    else:
        return 0

    eps = 1.
    phi = (2. / constants.pi) * atan(((2 * E_Morse) / (mass * vel**2)) ** 0.5)

    eps *= (cosh((1 + phi) * constants.pi * omega / (beta * vel))) ** 2

    eps *= 8 * ram**2 / (sinh(2 * constants.pi * omega / (beta * vel)))**2
    eps *= this_svt * (constants.pi ** 2) * omega * mass**2 / (osc_mass * (beta ** 2) * constants.h)

    if delta == 1:
        res = eps * exp(-(i + 1) * eps)
    elif delta == -1:
        res = eps * exp(-i * eps)
    return res


def vt_prob_g_only_fho(g: float, mass: float, beta: float, osc_mass: float,
                       ve_before: float, ve_after: float, i: int, delta: int,
                       ram1, ram2, E_Morse, this_svt) -> float:
    """
    Compute velocity-dependent part of
    VT transition probability, for heteronuclear molecules, compute 2 probabilities and take the average
    """
    if ram1 == ram2:
        return vt_prob_g_only_fho_12(g, mass, beta, osc_mass, ve_before, ve_after, i, delta, ram1, E_Morse, this_svt)
    else:
        res = vt_prob_g_only_fho_12(g, mass, beta, osc_mass, ve_before, ve_after, i, delta, ram1, E_Morse, this_svt)
        res += vt_prob_g_only_fho_12(g, mass, beta, osc_mass, ve_before, ve_after, i, delta, ram2, E_Morse, this_svt)
        return res / 2


def vt_rate_fho(T, beta, dref, omega, coll_mass, osc_mass,
                ve_before, ve_after,
                i, delta, ram1, ram2, E_Morse, Tref=273, this_svt=0.5):
    """
    Compute VT transition rate
    """
    if delta == 1:
        mult = (i + 1)
    elif delta == -1:
        mult = i
    elif delta > 0:
        mult = fact_div_fact(i, i + delta) / (factorial(delta) ** 2)
    else:
        mult = fact_div_fact(i + delta, i) / (factorial(-delta) ** 2)
    kT = T * constants.k
    mult *= (kT / (2.0 * constants.pi * coll_mass)) ** 0.5
    gref = (2 * constants.k * Tref / coll_mass)**0.5

    if ve_after <= ve_before:
        min_g = 0.
    else:
        min_g = ((ve_after - ve_before) / kT)**0.5
    f = lambda g: vt_prob_g_only_fho(g * (2 * kT / coll_mass)**0.5, coll_mass, beta, osc_mass, ve_before, ve_after,
                                     i, delta, ram1, ram2,
                                     E_Morse, this_svt) * (cs_vss(g * (2 * kT / coll_mass)**0.5,
                                                                  dref, gref, omega) * g**3 * exp(-g**2))
    return 8 * mult * integrate.quad(f, min_g, np.inf)[0]


def VT_integral(T, Tv, vibr_spectrum, beta, dref, omega, coll_mass, osc_mass,
                ram1, ram2, E_Morse, Tref=273, this_svt=0.5):
    """
    Compute the sum of integrals over VT transition cross-section
    """
    res = 0.
    dEsq = ((vibr_spectrum[1] - vibr_spectrum[0]) / (constants.k * T))**2
    rev_k_mult = exp((vibr_spectrum[0] - vibr_spectrum[1]) / (constants.k * T))
    kTv = constants.k * Tv
    for i in range(vibr_spectrum.shape[0] - 1):
        vtr = vt_rate_fho(T, beta, dref, omega, coll_mass, osc_mass,
                          vibr_spectrum[i + 1], vibr_spectrum[i], i + 1, -1, ram1, ram2, E_Morse, Tref, this_svt)
        tmp = dEsq * exp(-vibr_spectrum[i + 1] / kTv)
        tmp *= vtr
        res += tmp

        # v -> v + 1
        tmp = dEsq * exp(-vibr_spectrum[i] / kTv)
        tmp *= vtr * rev_k_mult
        res += tmp

    return res / (Zv(Tv, vibr_spectrum) * 8)


def make_harmonic_spectrum(theta_v, theta_D):
    """
    Create a numpy array of vibrational energies, given the characteristic vibrational temperature
    and dissociation energy (in Kelvins)
    """
    return constants.k * np.arange(0, theta_D - theta_v / 2, theta_v)


def compute(molecules, partners, T_min=200., T_max=25000., Tv_min=200., Tv_max=25000., dT=100., output=True,
            integral_only=True, pressure=101325):
    result = {}
    T_arr = np.linspace(T_min, T_max, 1 + int((T_max - T_min) / dT))
    Tv_arr = np.linspace(Tv_min, Tv_max, 1 + int((Tv_max - Tv_min) / dT))
    result['__T_points'] = T_arr.shape[0]
    result['__Tv_points'] = Tv_arr.shape[0]

    if output:
        print(' '.join((str(len(molecules)), 'molecules,', str(len(partners)), 'interaction partners for each molecule')))
        print(' '.join((str(T_arr.shape[0] * Tv_arr.shape[0]), 'points for each interaction')))

    for mol in molecules:
        v_spectrum = make_harmonic_spectrum(mol_data[mol]['theta_v'], mol_data[mol]['theta_D'])
        for p in partners:
            res = np.zeros((T_arr.shape[0], Tv_arr.shape[0]))
            for i, T in enumerate(T_arr):
                for j, Tv in enumerate(Tv_arr):
                    tmp = VT_integral(T, Tv, v_spectrum, FHO_data[mol + '+' + p]['beta'],
                                      vss_data[mol + '+' + p]['dref'],
                                      vss_data[mol + '+' + p]['omega'],
                                      red_masses[mol + '+' + p], mol_data[mol]['osc_mass'],
                                      ram_masses[mol][0], ram_masses[mol][1],
                                      FHO_data[mol + '+' + p]['E_Morse'], 273, FHO_data[mol + '+' + p]['svt'])
                    if not integral_only:
                        # if we return the whole relaxation time computed at a pressure of p Pa
                        tmp *= 4 * pressure / T / (masses[mol] * c_vibr(Tv, v_spectrum, masses[mol]))
                        tmp = 1. / tmp

                    res[i, j] = tmp
            result[mol + ',' + p] = np.copy(res)
            if output:
                print(mol, p, res[0, 0], res[-1, -1])

    return result


def write_csv(filename_prefix, molecules, partners, result, delimiter=",",
              T_min=200., T_max=25000., Tv_min=200., Tv_max=25000., dT=100.):

    header = "Temperature varies across rows; vib. temperature varies across columns; T_min=" + str(T_min)
    header += "; T_max=" + str(T_max) + "; Tv_min=" + str(Tv_min) + "; Tv_max=" + str(Tv_max) + "; dT=" + str(dT)
    for mol in molecules:
        for p in partners:
            np.savetxt('_'.join((filename_prefix, mol, p)) + '.csv', result[mol + ',' + p], delimiter=delimiter, newline='\n',
                       header=header)


def write_netcdf(filename, molecules, partners, result, format="NETCDF4_CLASSIC",
                 T_min=200., T_max=25000., Tv_min=200., Tv_max=25000., dT=100.):
    rootgrp = Dataset(filename + '.cdf', "w", format=format)
    rootgrp.createDimension("VT_data_nx", result['__T_points'])
    rootgrp.createDimension("VT_data_ny", result['__Tv_points'])
    rootgrp.VT_data_x_min = T_min
    rootgrp.VT_data_x_max = T_max
    rootgrp.VT_data_y_min = Tv_min
    rootgrp.VT_data_y_max = Tv_max
    rootgrp.VT_data_dx = dT
    rootgrp.VT_data_dy = dT

    for mol in molecules:
        for p in partners:
            vars_list = rootgrp.createVariable('VT_integral_table_' + mol + '_' + p, 'f8',
                                               ('VT_data_nx', 'VT_data_ny'))
            vars_list[:] = result[mol + ',' + p]
    rootgrp.close()


def main():
    parser = argparse.ArgumentParser(description='VT relaxation types calculation')
    parser.add_argument('-t','--outputfiletype' ,type=str, default='NETCDF4',
                        help='Output filetype: CSV or NETCDF4 (default is "NETCDF4"')
    parser.add_argument('-f','--outputfilename', type=str, default='VT_times', help="Output filename (or prefix in case of CSV files)")
    parser.add_argument('--cdfoutputfileformat', type=str, default='NETCDF4_CLASSIC',
                        help='For netCDF4 output, specifies output (default is "NETCDF4_CLASSIC")')
    parser.add_argument('--delimiter', type=str, default=',',
                        help='For CSV output, specifies delimiter (default is ",")')
    parser.add_argument('--molecules', type=str, default="N2,O2,NO",
                        help='Comma-separated names of molecules for which the VT relaxation times are computed (default is "N2,O2,NO")')
    parser.add_argument('--partners', type=str, default="N2,O2,NO,N,O,Ar",
                        help='Comma-separated names of particles, possible collision partners (default is "N2,O2,NO,N,O,Ar")')
    parser.add_argument('--temperaturemin', type=float, default=200.0, help='Minimum temperature (default is 200.0)')
    parser.add_argument('--temperaturemax', type=float, default=25000.0, help='Maximum temperature (default is 25000.0)')
    parser.add_argument('--vtemperaturemin', type=float, default=200.0, help='Minimum vibrational temperature (default is 200.0)')
    parser.add_argument('--vtemperaturemax', type=float, default=25000.0, help='Maximum vibrational temperature (default is 25000.0)')
    parser.add_argument('--dt', type=float, default=100.0, help='Temperature step size (default is 100.0)')
    parser.add_argument('--verbose', type=str, default="true",
                        help="If set to true (default value), will enable some output during computation")
    parser.add_argument('--integral_only', type=str, default="true",
                        help="If set to true (default value), will compute only averaging operator and not the full relaxation time")
    parser.add_argument('--pressure', type=float, default=101325,
                        help="If integral_only is not true, this will specify the pressure" +
                        " in Pascals at which the relaxation times are computed (default is 101325 Pa)")
    args = parser.parse_args()
    print(args)

    molecules = args.molecules.split(',')
    partners = args.partners.split(',')
    verbose_output = False
    integral_only = False
    if args.verbose == "true":
        verbose_output = True
    if args.integral_only == "true":
        integral_only = True

    res = compute(molecules, partners, T_min=args.temperaturemin, T_max=args.temperaturemax,
                  Tv_min=args.vtemperaturemin, Tv_max=args.vtemperaturemax, dT=args.dt, output=verbose_output,
                  integral_only=integral_only, pressure=args.pressure)

    if args.outputfiletype == "NETCDF4":
        write_netcdf(args.outputfilename, molecules, partners, res, args.cdfoutputfileformat,
                     T_min=args.temperaturemin, T_max=args.temperaturemax,
                     Tv_min=args.vtemperaturemin, Tv_max=args.vtemperaturemax, dT=args.dt)
    else:
        write_csv(args.outputfilename, molecules, partners, res, args.delimiter,
                  T_min=args.temperaturemin, T_max=args.temperaturemax,
                  Tv_min=args.vtemperaturemin, Tv_max=args.vtemperaturemax, dT=args.dt)


if __name__ == "__main__":
    main()
