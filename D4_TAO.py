import os
import numpy as np
import xarray as xr
import pandas as pd

abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
loc_bt = abspath + '/DataSet/X/FY3'
tao_loc = abspath + '/DataSet/Z/TAO1/SSW'
tao_path_list = sorted([tao_loc + '/' + i for i in os.listdir(tao_loc)])
# for orbit in ['asc', 'dsc']:
for orbit in ['dsc']:
    trainval_list = []
    bt_list = loc_bt + '/{0}'.format(orbit.capitalize())
    bt_list = sorted([bt_list + '/' + i for i in os.listdir(bt_list)])
    for file_bt in bt_list:
        if file_bt.split('_')[1] == orbit:
            trainval_list.append([orbit, '{0}'.format(file_bt.split('_')[-1])])
    val_list = []
    for data in trainval_list:
        if data[1].split('.')[0] in ['2022', '2023']:
            val_list.append(data)
    tao_path_list = tao_path_list[:14]
    for tao_path_num in range(len(tao_path_list)):
        if tao_path_list[tao_path_num].split('/')[-1] not in ['TAO_T8N165E_DM390A-20200830_D_WIND_10min_sub.nc']:
            tao_grid = []
            tao = xr.open_dataset(tao_path_list[tao_path_num])
            height = int(tao['HEIGHT'])
            lat = int(tao['LATITUDE'])
            lon = int(tao['LONGITUDE'])
            date_list = list(pd.to_datetime(tao['TIME']).strftime('%Y.%m.%d.%H.%M'))
            for bt in val_list:
                bt_year = bt[1].split('.')[0]
                bt_month = bt[1].split('.')[1]
                bt_day = bt[1].split('.')[2]
                bt_hour = bt[1].split('.')[3]
                bt_minute = bt[1].split('.')[4]
                exist = False
                for date_num in range(len(date_list)):
                    if exist == False:
                        tao_year = date_list[date_num].split('.')[0]
                        tao_month = date_list[date_num].split('.')[1]
                        tao_day = date_list[date_num].split('.')[2]
                        tao_hour = date_list[date_num].split('.')[3]
                        if (tao_year == bt_year) and (tao_month == bt_month) and (tao_day == bt_day) and (tao_hour == bt_hour):
                            bt_data = xr.open_dataset(loc_bt + '/{0}/FY3_{1}_bt_{2}'.format(bt[0].capitalize(), bt[0], bt[1]))
                            bt_lat = np.array(bt_data['lat']).astype(int)
                            bt_lon = np.array(bt_data['lon']).astype(int)
                            if len(np.where((bt_lat == lat) & (bt_lon == lon))[0]) != 0:
                                tao_grid.append(tao['WSPD'][:, 0].interp(TIME = '{0}-{1}-{2}T{3}:{4}'.format(bt_year, bt_month, bt_day, bt_hour, bt_minute)))
                                exist = True
            if tao_grid != []:
                tao_grid = xr.concat(tao_grid, dim = 'TIME')
                tao_grid = tao_grid.rename({'TIME':'time'})
                tao_grid = xr.Dataset({'tao':(('time'), np.array(tao_grid))}, coords = {'time':tao_grid['time'], 'lon':lon, 'lat':lat, 'height':height})['tao']
                tao_grid = tao_grid.astype(dtype = np.float32)
                tao_grid = xr.Dataset({'tao':tao_grid})
                tao_grid.to_netcdf(abspath + '/DataSet/Z/TAO/SSW/Grid_{0}_{1}'.format(orbit, tao_path_list[tao_path_num].split('/')[-1]))
            print('{0}, {1}/{2}'.format(orbit, tao_path_num + 1, len(tao_path_list)))

# tao_loc = abspath + '/DataSet/Z/TAO1/SST'
# tao_path_list = sorted([tao_loc + '/' + i for i in os.listdir(tao_loc)])
# for orbit in ['asc', 'dsc']:
#     trainval_list = []
#     bt_list = loc_bt + '/{0}'.format(orbit.capitalize())
#     bt_list = sorted([bt_list + '/' + i for i in os.listdir(bt_list)])
#     for file_bt in bt_list:
#         if file_bt.split('_')[1] == orbit:
#             trainval_list.append([orbit, '{0}'.format(file_bt.split('_')[-1])])
#     val_list = []
#     for data in trainval_list:
#         if data[1].split('.')[0] in ['2022', '2023']:
#             val_list.append(data)
#     for tao_path_num in range(len(tao_path_list)):
#         if tao_path_list[tao_path_num].split('/')[-1] not in ['TAO_T8N165E_DM390A-20200830_D_WIND_10min_sub.nc']:
#             tao_grid = []
#             tao = xr.open_dataset(tao_path_list[tao_path_num])
#             lat = int(tao['LATITUDE'])
#             lon = int(tao['LONGITUDE'])
#             date_list = list(pd.to_datetime(tao['TIME']).strftime('%Y.%m.%d.%H.%M'))
#             for bt in val_list:
#                 bt_year = bt[1].split('.')[0]
#                 bt_month = bt[1].split('.')[1]
#                 bt_day = bt[1].split('.')[2]
#                 bt_hour = bt[1].split('.')[3]
#                 bt_minute = bt[1].split('.')[4]
#                 exist = False
#                 for date_num in range(len(date_list)):
#                     if exist == False:
#                         tao_year = date_list[date_num].split('.')[0]
#                         tao_month = date_list[date_num].split('.')[1]
#                         tao_day = date_list[date_num].split('.')[2]
#                         tao_hour = date_list[date_num].split('.')[3]
#                         if (tao_year == bt_year) and (tao_month == bt_month) and (tao_day == bt_day) and (tao_hour == bt_hour):
#                             bt_data = xr.open_dataset(loc_bt + '/{0}/FY3_{1}_bt_{2}'.format(bt[0].capitalize(), bt[0], bt[1]))
#                             bt_lat = np.array(bt_data['lat']).astype(int)
#                             bt_lon = np.array(bt_data['lon']).astype(int)
#                             if len(np.where((bt_lat == lat) & (bt_lon == lon))[0]) != 0:
#                                 tao_grid.append(tao['SST'][:, 0].interp(TIME = '{0}-{1}-{2}T{3}:{4}'.format(bt_year, bt_month, bt_day, bt_hour, bt_minute)))
#                                 exist = True
#             if tao_grid != []:
#                 tao_grid = xr.concat(tao_grid, dim = 'TIME')
#                 tao_grid = tao_grid.rename({'TIME':'time'})
#                 tao_grid = xr.Dataset({'tao':(('time'), np.array(tao_grid))}, coords = {'time':tao_grid['time'], 'lon':lon, 'lat':lat})['tao']
#                 tao_grid = tao_grid.astype(dtype = np.float32)
#                 tao_grid = xr.Dataset({'tao':tao_grid})
#                 tao_grid.to_netcdf(abspath + '/DataSet/Z/TAO/SST/Grid_{0}_{1}'.format(orbit, tao_path_list[tao_path_num].split('/')[-1]))
#             print('{0}, {1}/{2}'.format(orbit, tao_path_num + 1, len(tao_path_list)))