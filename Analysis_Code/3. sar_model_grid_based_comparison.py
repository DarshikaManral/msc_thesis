from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import math
import csv
from pyproj import Geod, Proj
from math import sin, cos, radians, atan2, sqrt


# def getDistance(center_model, center_sar):
#     # approximate radius of earth in km
#     # R = 6373.0
#     R = 6372800
#     lat1 = radians(center_model[0])
#     lon1 = radians(center_model[1])
#     lat2 = radians(center_sar[0])
#     lon2 = radians(center_sar[1])
#
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * atan2(sqrt(a), sqrt(1 - a))
#
#     return R * c


def get_density_array(lon, lat, pixelsize_m):
    deltalat = pixelsize_m / 111000.0  # m to degrees
    deltalon = deltalat / np.cos(np.radians((lat_min + lat_max) / 2))
    lat_array = np.arange(lat_min - deltalat,
                          lat_max + deltalat, deltalat)
    lon_array = np.arange(lon_min - deltalat,
                          lon_max + deltalon, deltalon)
    bins = (lon_array, lat_array)

    H = np.zeros(shape=(len(lon_array) - 1, len(lat_array) - 1))  # .astype(int)

    weights = None
    H, dummy, dummy = \
        np.histogram2d(lon, lat,
                       weights=weights, bins=bins)

    return H


def get_deviation_angle():
    proj = Proj(proj='latlong', datum='WGS84')
    x0, y0 = proj(center_initial[1], center_initial[0])
    x1, y1 = proj(center_sar[1], center_sar[0])
    x2, y2 = proj(center_model[1], center_model[0])

    m2 = (y2 - y0) / (x2 - x0)
    m1 = (y1 - y0) / (x1 - x0)

    tan_theta_mean = abs((m2 - m1) / (1 + m1 * m2))

    return math.atan(tan_theta_mean)


def get_drift_angle(initial, final):
    proj = Proj(proj='latlong', datum='WGS84')

    x0, y0 = proj(initial[1], initial[0])

    north = [initial[1] + 0.001, initial[0]]
    x1, y1 = proj(north[1], north[0])

    x2, y2 = proj(final[1], final[0])

    dot = (x1 - x0) * (x2 - x0) + (y1 - y0) * (y2 - y0)  # dot product between [x1, y1] and [x2, y2]
    det = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)  # determinant
    radians = atan2(det, dot)
    degree = math.degrees(radians)

    if (degree > 0):
        angle = 360 - degree
    else:
        angle = -1 * degree

    return angle


def get_drift_direction(angle):
    if (0 <= angle <= 11.25) or (348.75 < angle <= 360):
        return 'N'
    elif 11.25 < angle <= 33.75:
        return 'NNE'
    elif 33.75 < angle <= 56.25:
        return 'NE'
    elif 56.25 < angle <= 78.75:
        return 'ENE'
    elif 78.75 < angle <= 101.25:
        return 'E'
    elif 101.25 < angle <= 123.75:
        return 'ESE'
    elif 123.75 < angle <= 146.25:
        return 'SE'
    elif 146.25 < angle <= 168.75:
        return 'SSE'
    elif 168.75 < angle <= 191.25:
        return 'S'
    elif 191.25 < angle <= 213.75:
        return 'SSW'
    elif 213.75 < angle <= 236.25:
        return 'SW'
    elif 236.25 < angle <= 258.75:
        return 'WSW'
    elif 258.75 < angle <= 281.25:
        return 'W'
    elif 281.25 < angle <= 303.75:
        return 'WNW'
    elif 303.75 < angle <= 326.25:
        return 'NW'
    elif 326.25 < angle <= 348.75:
        return 'NNW'


for i in range(03,4):  # 20160615_EM 20150610_EM60
    model_output = '/home/darshika/Desktop/NetCDF_outputs/Run{}_20110608_EM_F.nc'.format(str(i).zfill(2)) #20120615_EM
    print('Read for Model Output: ', model_output)
    dataset_model = Dataset(model_output, 'r')

    depth = dataset_model.variables['z']
    seed = depth.shape[0]
    print 'Number of seed points in Model scene: ', seed
    last_index = depth.shape[1] - 1
    final_depth = depth[:, last_index]

    steps = depth.shape[1]
    x_wind = dataset_model.variables['x_wind'][:]
    y_wind = dataset_model.variables['y_wind'][:]

    mean_x_wind = np.mean(x_wind[:])
    mean_y_wind = np.mean(y_wind[:])
    total_wind = round(math.sqrt(mean_x_wind ** 2 + mean_y_wind ** 2), 4)
    print total_wind

    # print 'Read for SAR scene'
    # if seed != 5000:
    sar_input = '/home/darshika/Desktop/NetCDF_outputs/0620_20120615_PO_{}.nc'.format(str(seed))
    # else:
    #     sar_input = '/home/darshika/Desktop/NetCDF_outputs/0615_20160615_EM.nc'
    dataset_sar = Dataset(sar_input, 'r')
    lons_sar = dataset_sar.variables['lon'][:, 0]
    lats_sar = dataset_sar.variables['lat'][:, 0]
    sar_seed = lons_sar.shape[0]
    print 'Number of seed points in Sar scene: ', sar_seed

    if sar_seed != seed:
       raise Exception("ERROR WRONG FILE COMPARISON")
    # Close the NC files- Not required now

    dataset_sar.close()

    # display seed points that are in the first 0.1m of the sea surface
    accepted_depth = depth[range(depth.shape[0]), last_index] > -0.1
    print 'Seed points count (below 0.1m)', sum(accepted_depth)

    lons_model = dataset_model.variables['lon'][:, last_index]
    lats_model = dataset_model.variables['lat'][:, last_index]

    lons_initial = dataset_model.variables['lon'][:, 0]
    lats_initial = dataset_model.variables['lat'][:, 0]

    print 'Number of seed points in model output: ', lons_model.shape[0]

    start_time = dataset_model.variables['time'][0]
    time_units = dataset_model.variables['time'].units
    start_timestamp = datetime(1970, 1, 1) + timedelta(seconds=int(start_time))

    end_time = dataset_model.variables['time'][last_index]
    end_timestamp = datetime(1970, 1, 1) + timedelta(seconds=int(end_time))

    dataset_model.close()

    # only keep the seed points with accepted depths
    lons2 = lons_model[accepted_depth == True]
    lats2 = lats_model[accepted_depth == True]

    # add buffer to the area
    lon_min = min(np.min(lons2), np.min(lons_sar)) - 0.005
    lat_min = min(np.min(lats2), np.min(lats_sar)) - 0.005
    lon_max = max(np.max(lons2), np.max(lons_sar)) + 0.005
    lat_max = max(np.max(lats2), np.max(lats_sar)) + 0.005

    # recommended for density distribution visualizations, however used here in this case too.
    # it is also a parameter - the values will differ if it is changed
    pixelsize_m = 50

    hist_model = get_density_array(lons2, lats2,
                                   pixelsize_m=pixelsize_m)
    hist_sar = get_density_array(lons_sar, lats_sar, pixelsize_m=pixelsize_m)

    print 'shape of histogram grid: ', hist_model.shape
    # 1: comparing non-zero grids - similar to area comparison

    print 'Non-Zero Model grids:', np.count_nonzero(hist_model)
    print 'Non -zero SAR grids:', np.count_nonzero(hist_sar)

    area_model = np.count_nonzero(hist_model) * pixelsize_m * pixelsize_m * 0.000001
    area_sar = np.count_nonzero(hist_sar) * pixelsize_m * pixelsize_m * 0.000001
    print 'Model Area km2:', area_model
    print 'SAR Area km2:', area_sar

    area_difference = abs(
        np.count_nonzero(hist_model) - np.count_nonzero(hist_sar)) * pixelsize_m * pixelsize_m * 0.000001

    print 'Difference in area in km2: ', area_difference

    # 2: METRIC replacing non-zero grids with value one and calculating the grid by grid difference

    hist_model[hist_model > 1] = 1
    hist_sar[hist_sar > 1] = 1

    print 'sum hist_model_binary:', np.sum(hist_model)
    print 'sum hist_sar_binary:', np.sum(hist_sar)

    # grid by grid difference
    difference = np.abs(np.subtract(hist_model, hist_sar))

    distribution_metric = np.sum(difference)
    print 'grid by grid difference: ', distribution_metric

    # 3. CENTER OF GRAVITY

    wgs84_geod = Geod(ellps='WGS84')

    # CG MODEL
    center_model = [np.mean(lats2), np.mean(lons2)]

    print 'Center of gravity- Model: ', center_model

    # CG SAR
    center_sar = [np.mean(lats_sar), np.mean(lons_sar)]
    print 'Center of gravity- SAR: ', center_sar

    # INITIAL
    center_initial = [np.mean(lats_initial), np.mean(lons_initial)]
    print 'Center of gravity- Initial: ', center_initial

    theta_mean = get_deviation_angle()
    print 'Theta: ', theta_mean

    angular_deviation_degrees = math.degrees(theta_mean)
    print 'Theta(degrees): ', angular_deviation_degrees

    #   4. Distance between Center of gravity of SAR and Model

    az1, az2, center_distance = wgs84_geod.inv(center_model[1], center_model[0], center_sar[1], center_sar[0])
    # center_distance = center_distance / 1000  # convert to km
    print "SAR-Model CG distance km: ", center_distance

    # dis = getDistance(center_model, center_sar)

    # region VELOCITY CALCULATIONS- SHORT TERM SIMULATIONS
    # velocity and distance in m/s
    #   5. Velocity of oil drift obtained from  : For Short term simulations only
    time_diff = (end_time - start_time)  # returns time difference in seconds

    # ************1)consecutive SAR scenes

    #   From Center of gravity
    # using library function
    az1, az2, sar_cg_distance = wgs84_geod.inv(center_initial[1], center_initial[0], center_sar[1], center_sar[0])
    # dis2 = getDistance(center_initial, center_sar)
    sar_cg_velocity = sar_cg_distance / time_diff
    sar_drift_angle = get_drift_angle(center_initial, center_sar)
    sar_drift_direction = get_drift_direction(sar_drift_angle)
    print "Distance b/w CG of Initial SAR and Final SAR: ", sar_cg_distance
    print "Velocity b/w CG of Initial SAR and Final SAR: ", sar_cg_velocity
    print "Angle from True North b/w CG of Initial SAR and Final SAR: ", sar_drift_angle
    print "Direction from True North b/w CG of Initial SAR and Final SAR: ", sar_drift_direction

    # **********2) between Initial SAR scene and model output
    #        From Center of gravity
    az1, az2, model_cg_distance = wgs84_geod.inv(center_initial[1], center_initial[0], center_model[1],
                                                 center_model[0])

    # dis3 = getDistance(center_initial, center_model)
    model_cg_velocity = model_cg_distance / time_diff
    model_drift_angle = get_drift_angle(center_initial, center_model)
    model_drift_direction = get_drift_direction(model_drift_angle)
    print "Distance b/w CG of Initial SAR and Model Output: ", model_cg_distance
    print "Velocity b/w CG of Initial SAR and Model Output: ", model_cg_velocity
    print "Angle from True North b/w CG of Initial SAR and Model output: ", model_drift_angle
    print "Direction from True North b/w CG of Initial SAR and Model output: ", model_drift_direction
    # endregion

    stats = [i, area_model, area_sar, area_difference, angular_deviation_degrees, center_distance / 1000,
             distribution_metric,
             sar_cg_distance / 1000, model_cg_distance / 1000, sar_cg_velocity, model_cg_velocity, sar_drift_angle,
             model_drift_angle, sar_drift_direction, model_drift_direction,
             center_initial, center_sar,
             center_model, model_output, sar_input]
    print stats
    #
    # stats= [total_wind]
    with open('/home/darshika/Desktop/Outputs/new_oft.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(stats)
    csvFile.close()
