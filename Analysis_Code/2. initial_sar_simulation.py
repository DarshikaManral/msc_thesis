#!/usr/bin/env python
import csv
from opendrift.readers import reader_current_from_drifter
from opendrift.models.openoil3D import OpenOil3D
from opendrift.readers import reader_ROMS_native, reader_netCDF_CF_generic
from datetime import datetime, timedelta
import numpy as np
import sys

# BULK RUNS
# CHECK OIL TYPE, START SHAPEFILE, DATE_TIME, IF PLANT OIL- DISABLE EMULSIFICATION

# to run from command line and pass parameters
time_step = int(sys.argv[1])
print time_step
time_step_output = int(sys.argv[2])
print time_step_output
mins = int(sys.argv[3])
print mins
seed_points = int(sys.argv[4])
print seed_points
water_fraction = float(sys.argv[5])
print water_fraction
oil_film_thickness_in_um = float(sys.argv[6])
print oil_film_thickness_in_um
output_file = sys.argv[7]
print output_file

# for emulsion type
# em_type = sys.argv[9]
# # em_type = 'E40'
# print em_type

# only for initial point simulations
radius = int(sys.argv[8])
print radius

release_lat = 60.045
release_lon = 2.34

# 2015
# EM40
# release_lat = 60.03551
# release_lon = 2.388947

# EM60 60.04331, 2.389167
# release_lat = 60.04331
# release_lon = 2.389167

# EM 80 60.05167,2.390764
# release_lat = 60.05167
# release_lon = 2.390764

# PO 60.02733, 2.3837
# release_lat = 60.02733
# release_lon = 2.3837

release_duration_in_mins = 5
release_rate = 4.8

# for long term simulation
days = 0
hours = 13

# oiltype = 'OSEBERG BLEND 2007'
# oiltype= 'TROLL, STATOIL'
# oiltype = 'AASGARD A 2003'
oiltype= 'MARINE GAS OIL 500 ppm S 2017'  #plant oil alias

o = OpenOil3D(weathering_model='noaa')
o.set_config('general:basemap_resolution', 'c')
o.set_config('processes:update_oilfilm_thickness', True)
o.fallback_values['land_binary_mask'] = 0  # always ocean
# ****** for plant oil**************
o.set_config('processes:emulsification', False)

start_time = datetime(2011, 06, 8, 4, 8)

# Add AROME wind for 2015 data
# norkyst1 = reader_netCDF_CF_generic.Reader('/home/darshika/Desktop/masks/Norkyst/2015/AromeWind_10June2015_MET.nc')

# Add Norkyst reader
norkyst1 = reader_ROMS_native.Reader('/home/darshika/Desktop/masks/Norkyst/2011/norkyst_8June2011.nc')
# norkyst2 = reader_netCDF_CF_generic.Reader('/home/darshika/Desktop/masks/Norkyst/2015/norkyst_10June2015_MET.nc')

print norkyst1
# print norkyst2

# # region DRIFTERS
drifter_file = '/home/darshika/Desktop/masks/Drifters/2011/code_2011Campaign.csv'
# drifter_file = '/home/darshika/Desktop/masks/Drifters/2015/sldmb_full.csv'

lat_center = release_lat
lon_center = release_lon
x_center, y_center = norkyst1.lonlat2xy(lon_center, lat_center)

drifterlats = []
drifterlons = []
driftertimes = []
with open(drifter_file, 'rb') as file1:
    file_reader = csv.reader(file1, delimiter=',')
    next(file_reader)
    for row in file_reader:
        drifterlons.append(float(row[4]))
        drifterlats.append(float(row[3]))
        driftertimes.append(datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S'))

drifterlats = drifterlats[::-1]
drifterlons = drifterlons[::-1]
driftertimes = driftertimes[::-1]

wind = np.zeros((len(drifterlats), 2))
for i in range(len(driftertimes)):
    if norkyst1.start_time <= driftertimes[i] <= norkyst1.end_time:
        model_wind = norkyst1.get_variables(requested_variables=['x_wind', 'y_wind'], time=driftertimes[i],
                                            x=x_center, y=y_center)
        wind[i] = [model_wind['x_wind'][0][0], model_wind['y_wind'][0][0]]
    else:
        wind[i][0] = wind[i][1] = 0

drifter = reader_current_from_drifter.Reader(
    lons=drifterlons, lats=drifterlats, times=driftertimes, wind=wind)
o.add_reader(drifter)

# # endregion

# # order of reader is important to give priority to the source
# o.add_reader(norkyst2)
o.add_reader([norkyst1])

o.seed_elements(lon=release_lon, lat=release_lat, number=seed_points,
                radius=[radius],
                time=[start_time, start_time + timedelta(minutes=release_duration_in_mins)],
                oiltype=oiltype,
                water_fraction=water_fraction,
                m3_per_hour=release_rate)

o.run(time_step=time_step * 60, time_step_output=time_step_output * 60,
      duration=timedelta(days=days, hours=hours, minutes=mins),
      outfile='/home/darshika/Desktop/NetCDF_outputs/' + output_file + '.nc')

o.plot_oil_budget()
o.plot()
# o.animation()
