import os
import datetime
import numpy as np
import xarray as xr

abspath = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
begin_date = '2015-01-01'
end_date = '2019-12-31'
date_list = []
begin_datetime = datetime.datetime.strptime(begin_date, '%Y-%m-%d')
end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
while begin_datetime <= end_datetime:
    date_str = begin_datetime.strftime('%Y-%m-%d')
    date_list.append(date_str)
    begin_datetime = begin_datetime + datetime.timedelta(days = 1)
num = 6
interval = int(len(date_list) / 12) + 1
date_list = date_list[int(interval * (num - 1)):int(interval * num)]
lon, lat = np.meshgrid(np.arange(-179.75, 180.01, 0.25), np.arange(90, -90.01, -0.25))
lon = lon.flatten()
lat = lat.flatten()
points = np.concatenate([lon.reshape(-1, 1), lat.reshape(-1, 1)], axis = 1)
loc_era5 = 'F:/ERA5/'
loc_fy3c = abspath + '/DataSet/X/FY3C'
for orbit in ['asc', 'dsc']:
    file_list = loc_fy3c + '/{0}'.format(orbit.capitalize())
    file_list = sorted([file_list + '/' + i for i in os.listdir(file_list)])
    for year in range(int(begin_date.split('-')[0]), int(end_date.split('-')[0]) + 1):
        ussw = xr.open_dataset(loc_era5 + '/UWND10/ERA5_uwnd10_{0}.nc'.format(year))['u10']
        vssw = xr.open_dataset(loc_era5 + '/VWND10/ERA5_vwnd10_{0}.nc'.format(year))['v10']
        sst = xr.open_dataset(loc_era5 + '/SST/ERA5_sst_{0}.nc'.format(year))['sst']
        ussw['longitude_adjusted'] = xr.where(ussw['longitude'] > 180, ussw['longitude'] - 360, ussw['longitude'])
        ussw = (ussw.swap_dims({'longitude':'longitude_adjusted'}).sel(**{'longitude_adjusted':sorted(ussw.longitude_adjusted)}).drop('longitude'))
        ussw = ussw.rename({'longitude_adjusted':'longitude'})
        vssw['longitude_adjusted'] = xr.where(vssw['longitude'] > 180, vssw['longitude'] - 360, vssw['longitude'])
        vssw = (vssw.swap_dims({'longitude':'longitude_adjusted'}).sel(**{'longitude_adjusted':sorted(vssw.longitude_adjusted)}).drop('longitude'))
        vssw = vssw.rename({'longitude_adjusted':'longitude'})
        sst['longitude_adjusted'] = xr.where(sst['longitude'] > 180, sst['longitude'] - 360, sst['longitude'])
        sst = (sst.swap_dims({'longitude':'longitude_adjusted'}).sel(**{'longitude_adjusted':sorted(sst.longitude_adjusted)}).drop('longitude'))
        sst = sst.rename({'longitude_adjusted':'longitude'})
        for file in file_list:
            if (file.split('_')[1] == orbit) and (file.split('_')[-1][:4] == str(year)):
                month = file.split('.')[-5]
                day = file.split('.')[-4]
                hour = file.split('.')[-3]
                minute = file.split('.')[-2]
                if '{0}-{1}-{2}'.format(year, month, day) in date_list:
                    save_path = abspath + '/DataSet/Y/ERA5/{0}/ERA5_{1}_sswsst_{2}.{3}.{4}.{5}.{6}.nc'.format(orbit.capitalize(), orbit, year, month, day, hour, minute)
                    sswsst_grid = []
                    if os.path.exists(save_path) == False:
                        fy3c = xr.open_dataset(file)
                        bt = fy3c['bt']
                        lat_grid = np.array(fy3c['lat'])
                        lon_grid = np.array(fy3c['lon'])
                        lat_grid = np.trunc(lat_grid / 0.25) * 0.25
                        lon_grid = np.trunc(lon_grid / 0.25) * 0.25
                        lon_grid[np.where(lon_grid == -180)] = -179.75
                        lat_grid_idx = -(lat_grid - 90) * 4
                        lon_grid_idx = (lon_grid + 179.75) * 4
                        lat_grid_idx = lat_grid_idx.astype(dtype = int)
                        lon_grid_idx = lon_grid_idx.astype(dtype = int)
                        for sswsst_str in ['ussw', 'vssw', 'sst']:
                            if sswsst_str == 'ussw':
                                sswsst = ussw.interp(time = '{0}-{1}-{2}T{3}:{4}'.format(year, month, day, hour, minute))
                            elif sswsst_str == 'vssw':
                                sswsst = vssw.interp(time = '{0}-{1}-{2}T{3}:{4}'.format(year, month, day, hour, minute))
                            elif sswsst_str == 'sst':
                                sswsst = sst.interp(time = '{0}-{1}-{2}T{3}:{4}'.format(year, month, day, hour, minute))
                            sswsst = np.array(sswsst)[lat_grid_idx, lon_grid_idx]
                            sswsst_grid.append(sswsst)
                        sswsst_grid = np.array(sswsst_grid)
                        sswsst_grid = sswsst_grid.astype(dtype = np.float32)
                        lat_grid = lat_grid.astype(dtype = np.float32)
                        lon_grid = lon_grid.astype(dtype = np.float32)
                        output = xr.Dataset({'sswsst':(['var', 'sswsst_num'], sswsst_grid), 'lat':(['sswsst_num'], lat_grid), 'lon':(['sswsst_num'], lon_grid)}, coords = {'var':['ussw', 'vssw', 'sst'], 'sswsst_num':np.arange(len(lat_grid))})
                        output.to_netcdf(save_path)
                        print('ERA5_{0}_{1}.{2}.{3}.{4}.{5}'.format(orbit, year, month, day, hour, minute))