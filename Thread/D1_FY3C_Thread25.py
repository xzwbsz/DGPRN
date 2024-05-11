import os
import datetime
import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.interpolate import griddata

abspath = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
begin_date = '2018-01-01'
end_date = '2018-12-31'
date_list = []
begin_datetime = datetime.datetime.strptime(begin_date, '%Y-%m-%d')
end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
while begin_datetime <= end_datetime:
    date_str = begin_datetime.strftime('%Y-%m-%d')
    date_list.append(date_str)
    begin_datetime = begin_datetime + datetime.timedelta(days = 1)
num = 25
interval = int(len(date_list) / 25) + 1
date_list = date_list[int(interval * (num - 1)):int(interval * num)]
loc = 'I:/FY3C/MWRI'
error = 'Y'
for orbit in ['asc', 'dsc']:
    for year in range(int(begin_date.split('-')[0]), int(end_date.split('-')[0]) + 1):
        file_list = loc + '/{0}/{1}'.format(year, orbit.capitalize())
        file_list = sorted([file_list + '/' + i for i in os.listdir(file_list)])
        for file in file_list:
            if file.split('_')[-1] == 'MS.HDF':
                month = file.split('_')[4][4:6]
                day = file.split('_')[4][6:]
                hour = file.split('_')[5][:2]
                minute = file.split('_')[5][2:]
                if '{0}-{1}-{2}'.format(year, month, day) in date_list:
                    save_path = abspath + '/DataSet/X/FY3C/{0}/FY3C_{1}_bt_{2}.{3}.{4}.{5}.{6}.nc'.format(orbit.capitalize(), orbit, year, month, day, hour, minute)
                    if os.path.exists(save_path) == False:
                        bt_all = []
                        try:
                            file = nc.Dataset(file)
                            bt = file.groups['Data']['EARTH_OBSERVE_BT_10_to_89GHz']
                            lat = file.groups['Geolocation']['Latitude']
                            lon = file.groups['Geolocation']['Longitude']
                            mask = file.groups['Data']['LandSeaMask']
                            for channel_num in range(10):
                                bt_all.append(np.array(bt[channel_num]).flatten())
                            lat = np.array(lat).flatten()
                            lon = np.array(lon).flatten()
                            mask = np.array(mask).flatten()
                            for channel_num in range(10):
                                bt_all[channel_num] = np.delete(bt_all[channel_num], np.where(mask != 3), axis = 0)
                            lat = np.delete(lat, np.where(mask != 3), axis = 0)
                            lon = np.delete(lon, np.where(mask != 3), axis = 0)
                            lat = np.delete(lat, np.where(bt_all[-1] == 29999), axis = 0)
                            lon = np.delete(lon, np.where(bt_all[-1] == 29999), axis = 0)
                            for channel_num in range(10):
                                bt_all[channel_num] = np.delete(bt_all[channel_num], np.where(bt_all[-1] == 29999), axis = 0)
                            for channel_num in range(10):
                                bt_all[channel_num] = np.delete(bt_all[channel_num], np.where(lat == 999.9), axis = 0)
                            lon = np.delete(lon, np.where(lat == 999.9), axis = 0)
                            lat = np.delete(lat, np.where(lat == 999.9), axis = 0)
                            for channel_num in range(10):
                                bt_all[channel_num] = np.delete(bt_all[channel_num], np.where(lon == 999.9), axis = 0)
                            lat = np.delete(lat, np.where(lon == 999.9), axis = 0)
                            lon = np.delete(lon, np.where(lon == 999.9), axis = 0)
                            bt_all = np.array(bt_all)
                            lat_all = np.array(lat)
                            lon_all = np.array(lon)
                            points = np.concatenate([lon_all.reshape(-1, 1), lat_all.reshape(-1, 1)], axis = 1)
                            points_grid = []
                            for points_num in range(len(points)):
                                lat = round(lat_all[points_num] / 0.125) * 0.125
                                lon = round(lon_all[points_num] / 0.125) * 0.125
                                points_grid.append([lat, lon])
                            if len(points_grid) != 0:
                                points_grid = [list(i) for i in set(tuple(_) for _ in points_grid)]
                                points_grid = np.array(points_grid)
                                lat_grid = points_grid[:, 0]
                                lon_grid = points_grid[:, 1]
                                bt_grid = np.zeros([10, len(points_grid)])
                                for channel_num in range(10):
                                    bt_grid[channel_num] = griddata(points, bt_all[channel_num].reshape(-1, 1), (lon_grid.reshape(-1, 1), lat_grid.reshape(-1, 1)), method = 'linear')[:, :, 0][:, 0]
                                    bt_grid[channel_num] = bt_grid[channel_num] * 0.01 + 327.68
                                bt_grid = bt_grid.astype(dtype = np.float32)
                                lat_grid = lat_grid.astype(dtype = np.float32)
                                lon_grid = lon_grid.astype(dtype = np.float32)
                                output = xr.Dataset({'bt':(['channel_num', 'bt_num'], bt_grid), 'lat':(['bt_num'], lat_grid), 'lon':(['bt_num'], lon_grid)}, coords = {'channel_num':np.arange(10), 'bt_num':np.arange(len(lat_grid))})
                                output.to_netcdf(save_path)
                                print('FY3C_{0}_{1}.{2}.{3}.{4}.{5}'.format(orbit, year, month, day, hour, minute))
                        except:
                            error = 'N'
                            print('ERROR-FY3C_{0}_{1}.{2}.{3}.{4}.{5}'.format(orbit, year, month, day, hour, minute))
print(error)