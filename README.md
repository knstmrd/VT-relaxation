# VT relaxation time computation

Python code to compute VT relaxation times using a kinetic theory based definition, as developed in [Kustova, E. V., and G. P. Oblapenko. "Mutual effect of vibrational relaxation and chemical reactions in viscous multitemperature flows." Physical Review E, 2016](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.93.033127) and [G. P. Oblapenko. "Calculation of Vibrational Relaxation Times Using a Kinetic Theory Approach." The Journal of Physical Chemistry A, 2018](http://doi.org/10.1021/acs.jpca.8b09897).

More specifically, it computes the either the averaging operator over all possible one-quantum VT transitions using the [Forced Harmonic Oscillator model](https://arc.aiaa.org/doi/abs/10.2514/2.6302) and the [Variable Soft Sphere model with recent collision parameters](https://aip.scitation.org/doi/abs/10.1063/1.4939719) to calculate the transition cross-sections.

The code can compute either the averaging operator `avg_E_vibr`, or the full relaxation time `t` (at a specified pressure), defined as

`t = m * c_vibr / (4 * k * n * avg_E_vibr)`,

where `m` is the relaxating molecules mass, `k` is the Boltzmann constant, `n` is the number density, `c_vibr` is the specific heat of vibrational degrees of freedom.

# Usage


The code requires Python 3, the netCDF4 library, numpy and scipy.

To install the pre-requisites:

`pip install netCDF4`

`pip install numpy`

`pip install scipy`


The script can be run with the following optional arguments:

optional arguments:
- `-h, --help`: show this help message and exit
- `-t OUTPUTFILETYPE, --outputfiletype OUTPUTFILETYPE` specifies output filetype: CSV or NETCDF4 (default is "NETCDF4")
- `-f OUTPUTFILENAME, --outputfilename OUTPUTFILENAME` specifies output filename (or prefix in case of CSV files, a separate CSV file is created for each interaction pair)
- `--cdfoutputfileformat CDFOUTPUTFILEFORMAT` For netCDF4 output, specifies output format (default is "NETCDF4_CLASSIC")
- `--delimiter DELIMITER` For CSV output, specifies delimiter (default is ",")
- `--molecules MOLECULES` Comma-separated names of molecules for which the VT relaxation times are computed (default is "N2,O2,NO")
- `--partners PARTNERS   Comma-separated names of particles, possible collision partners (default is "N2,O2,NO,N,O,Ar")`
- `--temperaturemin TEMPERATUREMIN` Minimum temperature (default is 200.0)
- `--temperaturemax TEMPERATUREMAX` Maximum temperature (default is 25000.0)
- `--vtemperaturemin VTEMPERATUREMIN` Minimum vibrational temperature (default is 200.0)
- `--vtemperaturemax VTEMPERATUREMAX` Maximum vibrational temperature (default is 25000.0)
- `--dt DT` Temperature step size (default is 100.0)
- `--verbose VERBOSE`  If set to true (default value), will enable some output during computation
- `--integral_only INTEGRAL_ONLY` If set to true (default value), will compute only averaging operator and not the full relaxation time
- `--pressure PRESSURE` If integral_only is not true, this will specify the pressure in Pascals at which the relaxation times are computed (default is 101325 Pa)

# Citing

Oblapenko, G. P. Calculation of Vibrational Relaxation Times Using a Kinetic Theory Approach. The Journal of Physical Chemistry A, 2018. http://doi.org/10.1021/acs.jpca.8b09897

Kustova, E. V., and G. P. Oblapenko. Mutual effect of vibrational relaxation and chemical reactions in viscous multitemperature flows. Physical Review E, 2016. https://journals.aps.org/pre/abstract/10.1103/PhysRevE.93.033127