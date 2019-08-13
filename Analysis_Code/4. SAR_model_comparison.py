from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import basemap
from datetime import datetime, timedelta
import math


def polar_stere(lon_w, lon_e, lat_s, lat_n, **kwargs):
    '''Returns a Basemap object (NPS/SPS) focused in a region.

    lon_w, lon_e, lat_s, lat_n -- Graphic limits in geographical coordinates.
                                  W and S directions are negative.
    **kwargs -- Aditional arguments for Basemap object.

    '''
    lon_0 = lon_w + (lon_e - lon_w) / 2.
    ref = lat_s if abs(lat_s) > abs(lat_n) else lat_n
    lat_0 = math.copysign(90., ref)
    proj = 'npstere' if lat_0 > 0 else 'spstere'
    prj = basemap.Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0,
                          boundinglat=0, resolution='c')
    # prj = pyproj.Proj(proj='stere', lon_0=lon_0, lat_0=lat_0)
    lons = [lon_w, lon_e, lon_w, lon_e, lon_0, lon_0]
    lats = [lat_s, lat_s, lat_n, lat_n, lat_s, lat_n]
    x, y = prj(lons, lats)
    ll_lon, ll_lat = prj(min(x), min(y), inverse=True)
    ur_lon, ur_lat = prj(max(x), max(y), inverse=True)
    return basemap.Basemap(projection='stere', lat_0=lat_0, lon_0=lon_0,
                           llcrnrlon=ll_lon, llcrnrlat=ll_lat,
                           urcrnrlon=ur_lon, urcrnrlat=ur_lat, **kwargs)


print('Read for Model Output')
model_output = '/home/darshika/Desktop/NetCDF_outputs/Run03_20110608_EM_F.nc'

dataset_model = Dataset(model_output, 'r')
lons_model = dataset_model.variables['lon'][:]
lats_model = dataset_model.variables['lat'][:]

depth = dataset_model.variables['z']
print(depth.shape)
last_index = depth.shape[1] - 1

print('Number of Points: ', lons_model.shape[0])

time = dataset_model.variables['time'][last_index]
time_units = dataset_model.variables['time'].units
timestamp = datetime(1970, 1, 1) + timedelta(seconds=int(time))

model_depth = depth[range(depth.shape[0]), last_index]

# display seed points that are in the first 0.1m of the sea surface
accepted_depth = model_depth > -0.1
print 'Seed points count (below 0.1m)', sum(accepted_depth)

temp = depth[range(depth.shape[0]), last_index] * accepted_depth.astype(int)
unique = np.unique(temp)
print 'Unique count', len(unique)
print unique

print 'Read for SAR scene'
sar_input = '/home/darshika/Desktop/NetCDF_outputs/0623_20110608_EM_5000.nc'
dataset_sar = Dataset(sar_input, 'r')
lons_sar = dataset_sar.variables['lon'][:]
lats_sar = dataset_sar.variables['lat'][:]
print('Number of Points: ', lons_sar.shape[0])

# Close the NC files- Not required now
dataset_model.close()
dataset_sar.close()

#### for comparison with another oil type
#
# sar_oiltype2 = '/home/darshika/Desktop/NetCDF_outputs/test_20150610_troll.nc'
# dataset_oiltype2 = Dataset(sar_oiltype2, 'r')
# lons_oiltype2 = dataset_oiltype2.variables['lon'][:]
# lats_oiltype2 = dataset_oiltype2.variables['lat'][:]
#
# depth_oiltype2 = dataset_oiltype2.variables['z']
# print(depth_oiltype2.shape)
#
# oiltype2_depth = depth_oiltype2[range(depth_oiltype2.shape[0]), last_index]
# accepted_oiltype2_depth = oiltype2_depth >- 0.1
# dataset_oiltype2.close()

# add buffer to the area
lon_min = min(lons_model.min(), lons_sar.min()) - 0.02
lat_min = min(lats_model.min(), lats_sar.min()) - 0.01
lon_max = max(lons_model.max(), lons_sar.max()) + 0.02
lat_max = max(lats_model.max(), lats_sar.max()) + 0.01
# lon_min = 2.37
# lat_min = 59.99
# lon_max = 2.66
# lat_max = 60.04
print 'Preparing basemap...'

# Basemap = basemap.Basemap
# m = Basemap(llcrnrlon=lon_min,
#             llcrnrlat=lat_min,
#             urcrnrlon=lon_max,
#             urcrnrlat=lat_max,
#             lat_0=(lat_max - lat_min)/2,
#             lon_0=(lon_max-lon_min)/2,
#             projection='merc',
#             resolution= 'c')

m = polar_stere(lon_min, lon_max, lat_min, lat_max)

print'Plotting points on the basemap...'
m.scatter(lons_model[range(lons_model.shape[0]), 1],
          lats_model[range(lons_model.shape[0]), 1],
          s=1,
          edgecolor=None,
          color='lightgrey',
          # linewidths=.2,
          latlon=True,
          label='Initial SAR')

# Original
# m.scatter(lons[range(lons.shape[0]), last_index],
#           lats[range(lons.shape[0]), last_index],
#           s=1,
#           edgecolor=None,
#           color='lightgrey',
#           # linewidths=.2,
#           latlon=True,
#           label='Model',
#           alpha = 0.4)

# plot only those points within 0.1m of the surface
m.scatter(lons_model[range(lons_model.shape[0]), last_index] * accepted_depth,
          lats_model[range(lons_model.shape[0]), last_index] * accepted_depth,
          s=1,
          edgecolor=None,
          color='orange',
          # linewidths=.2,
          latlon=True,
          label='Model',
          # label='Oseberg',
          alpha=0.4)

# m.scatter(lons_oiltype2[range(lons_oiltype2.shape[0]), last_index] * accepted_oiltype2_depth,
#           lats_oiltype2[range(lons_oiltype2.shape[0]), last_index] * accepted_oiltype2_depth,
#           s=1,
#           edgecolor=None,
#           color='limegreen',
#           label='Troll, Statoil',
#           #           c=oiltype2_depth,
#           #           cmap='Wistia',
#           latlon=True)

# plot with depth gradient
# m.scatter(lons_model[range(lons_model.shape[0]), last_index],
#           lats_model[range(lons_model.shape[0]), last_index],
#           c=model_depth,
#           s=1,
#           #edgecolor=None,
#           cmap='Blues',
#           # color='lightgrey',
#           # linewidths=.2,
#           latlon=True,
#            label='Oseberg')
#           #alpha = 0.4)


m.scatter(lons_sar[range(lons_sar.shape[0]), 0],
          lats_sar[range(lons_sar.shape[0]), 0],
          s=1,  # area
          edgecolor=None,
          color='turquoise',  # 'lightseagreen',
          # linewidths=.2,
          latlon=True,
          label='Final SAR')
# alpha=0.3)

# Center initial, center SAR, center model
# [60.065357, 2.3960347]	[60.0657, 2.3951821]	[60.067413, 2.3938906]
# [60.0472, 2.3986728]	[60.059414, 2.4380736]	[60.01695, 2.4759855]
# [60.0472, 2.3986728]	[60.059414, 2.4380736]	[60.028095, 2.5407224]
# initial
# m.scatter(np.mean(lons_model[range(lons_model.shape[0]), 0]),
#           np.mean(lats_model[range(lons_model.shape[0]), 0]),
#           s=10,  # area
#           edgecolor=None,
#           color='black',
#           latlon=True)
# # 2nd SAR
# m.scatter(np.mean(lons_sar[range(lons_sar.shape[0]), 0]),
#           np.mean(lats_sar[range(lons_sar.shape[0]), 0]),
#           s=10,  # area
#           edgecolor=None,
#           color='blue',
#           latlon=True)
# model
# temp1 = np.mean(lons_model[range(lons_model.shape[0]), last_index] * accepted_depth)
# temp2 = np.mean(lats_model[range(lons_model.shape[0]), last_index] * accepted_depth)
# print temp1,temp2
# m.scatter(np.mean(lons_model[range(lons_model.shape[0]), last_index] * accepted_depth),
#           np.mean(lats_model[range(lons_model.shape[0]), last_index] * accepted_depth),
#           s=10,  # area
#           edgecolor=None,
#           color='red',
#           latlon=True)

# large maps
m.drawmapscale(lon_max - 0.03, lat_min + 0.01, lons_model.mean(), lats_model.mean(), 2, barstyle='simple',
               labelstyle='simple')
# small maps
# m.drawmapscale(lon_min + 0.015, lat_min + 0.0025, lons_model.mean(), lats_model.mean(), 1, barstyle='simple',
#                labelstyle='simple')

# m.scatter(2.5, 60, color='r',s=10, latlon=True)
m.drawparallels(np.arange(-80., 81., 0.05), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 0.1), labels=[0, 0, 0, 1], fontsize=10)
m.drawcoastlines()

plt.title(timestamp.strftime("%d-%b-%Y (%H:%M:%S)"))
# plt.legend(markerscale=5)
plt.show()
