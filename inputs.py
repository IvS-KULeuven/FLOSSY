from astropy import units as u

# Set TIC number of your TESS star
TIC = 374944608
outputDir = './outputs'

# Set path containing the prewhitened frequencies. Must be a CSV file with header.
pw_file = f'pw/pw_tic{TIC}.csv'
# Names in your CSV file of the 3 column names used by the code ('frequency', 'frequency_error' and 'amplitude').
pw_col_names = {'frequency': 'frequency',
                'frequency_error': 'e_frequency',
                'amplitude': 'amp'}
# Set unis for the 3 columns. Use astropy units. (See examples at: https://docs.astropy.org/en/stable/units/index.html#module-astropy.units.si)
pw_units = {'frequency': 1/u.day, # 1/day
            'frequency_error': 1/u.day, # 1/day
            'amplitude': 1e-3*u.dimensionless_unscaled} # ppt
# Set the frequency resolution. Use astropy units.
pw_frequency_resolution = 1/(365*u.day)

# Set path containing the periodogram. Must be a CSV file with header.
pg_file = f'pg/pg_tic{TIC}.csv'
# Names in yout CSV file of the 2 column names used by the code ('frequency' and 'amplitude').
pg_col_names = {'frequency': 'freq',
                'amplitude': 'amp'}
# Set unis for the 2 columns. Use astropy units. (See examples at: https://docs.astropy.org/en/stable/units/index.html#module-astropy.units.si)
pg_units = {'frequency': 1/u.day, # 1/day
            'amplitude': 1e-3*u.dimensionless_unscaled} # ppt

# Set plots units 
# plots_units = {'period': u.day, # day
#                'period spacing': 100*u.s, # kilo-second
#                'amplitude': 1e-3*u.dimensionless_unscaled} # ppt
