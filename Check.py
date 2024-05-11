import os
import torch
import cmaps
import random
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import scipy.sparse as sp
import cartopy.crs as ccrs
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from subprocess import Popen
from matplotlib import rcParams
from torch_geometric.data import Data
from scipy.interpolate import griddata
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# os.system('start cmd')
# os.system('start cmd /k'+ 'ls')
# os.system("start cmd.exe /k conda activate python310 && python")
# os.system("start powershell.exe cmd /k conda activate python310 && python")
# thread = 5
# run_command = ''
# for thread_num in range(1, thread + 1):
#     run_command = run_command + 'python E:\DATA\Python\SMV\Code\Thread\D1_FY3C_Thread{0}.py & '.format(thread_num)
# run_command = run_command[:-3]
# print(run_command)

# loc = 'E:/DATA/Python/SMV/DataSet/X/FY3C/ASD_2018'
# file_list = os.listdir(loc)
# for file in file_list:
#     os.rename(loc + '/' + file, loc + '/' + file.replace('ASD_bt', 'asc_bt'))

# Popen('start cmd.exe && cd', shell = True)
# Popen('b.exe', shell=False)

# print(np.arange(-179.75, 180.01, 0.25))

# a = np.zeros([2, 80])
# a[0] = np.arange(10, 20, 0.125)
# a[1] = np.arange(20, 30, 0.125)
# print(a[[2, 3], [2, 3]])

# a = np.arange(10, 20, 0.125)
# print(a[::2])
# print(a[[2, 2]])

# a = [1, 2, 3]
# b = ['a']
# print(a + b)

# data_idx = np.arange(1441 * 2881).reshape(1441, 2881)
# print(data_idx[0])

# mask = 1 != 2
# print(mask)
 
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
# print(x.shape)
# print(edge_index.shape)
# data = Data(x=x, edge_index=edge_index)
# print(data)
# print(data.x)
# >>> Data(edge_index=[2, 4], x=[3, 1])

# a = 1e-4
# print(a)

# a = [3, 1, 2, 2, 2, 5]
# a = torch.tensor(a)
# # a = np.argsort(np.argsort(a))
# a = torch.unique(a, return_inverse = True)[1]
# print(a)

# edge_idx = edge_idx.cpu().numpy()
# # print([((edge_idx[:1000, 1] % 2881) / 8) - 180 , -((edge_idx[:1000, 1] // 2881) / 8) + 90])
# fig = plt.figure(figsize = (8, 8))
# ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
# ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree())
# # ax.set_xticks(np.arange(lon[0] - 90 + 5, lon[-1] - 90, 20))
# # ax.set_yticks(np.arange(lat[-1], lat[0] + 0.1, 20))
# ax.xaxis.set_major_formatter(LongitudeFormatter())
# ax.yaxis.set_major_formatter(LatitudeFormatter())
# ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
# # pcolor = plt.scatter(lon_grid, lat_grid, bt, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18)
# pcolor = plt.scatter(np.array(idx_bt_data['lon'][fillnan_idx]), np.array(idx_bt_data['lat'][fillnan_idx]), c = np.array(idx_bt_data['bt'])[3, fillnan_idx], transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18, s = 1)
# start_lon = ((edge_idx[:, 0] % 2881) / 8) - 180
# start_lat = -((edge_idx[:, 0] // 2881) / 8) + 90
# end_lon = ((edge_idx[:, 1] % 2881) / 8) - 180
# end_lat = -((edge_idx[:, 1] // 2881) / 8) + 90
# # for n in range(10000):
# #     plt.plot([start_lon[n], end_lon[n]], [start_lat[n], end_lat[n]], c = 'red', transform = ccrs.PlateCarree())
# colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'm/s')
# plt.show()
# # plt.savefig(abspath + '/Save/Plot/Test/Edge_{0}.png'.format(idx), dpi = 1000, bbox_inches = 'tight')

# print(round(1.5))

# array2a = np.array([[1, 2], [3, 3], [2, 1], [1, 3], [2, 1]])
# array2b = np.array([[2, 1], [1, 4], [3, 3]])
# test = array2a[:, None] == array2b
# print(test)
# print(array2b[np.all(test.mean(0) > 0, axis = 1)]) # [[2 1] # [3 3]]

# print(np.array([[1, 2], [3, 3], [2, 1], [1, 3], [2, 1]]).shape)

# print(list(range(10))[:4])

# file_list = ['2018.05.07.05.41.nc', '2018.12.25.01.34.nc', '2018.12.25.03.16.nc']
# idx_bt_data = xr.open_dataset('/home/gogooz/Python/SWS/DataSet/X/FY3C/Dsc/FY3C_dsc_bt_{0}'.format(file_list[0]))
# idx_bt = np.array(idx_bt_data['bt'])
# idx_lat = np.array(idx_bt_data['lat'])
# idx_lon = np.array(idx_bt_data['lon'])
# # slice_idx = np.where((idx_lat >= 1) & (idx_lat <= 10) & (idx_lon >= 1) & (idx_lon <= 10))[0]
# slice_idx = np.where((idx_lat >= 1) & (idx_lat <= 10))[0]
# idx_bt = idx_bt[:, slice_idx]
# idx_lat = idx_lat[slice_idx]
# idx_lon = idx_lon[slice_idx]
# fig = plt.figure(figsize = (8, 8))
# ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
# ax.set_extent([int(np.min(idx_lon)), int(np.max(idx_lon)), 1, 10], crs = ccrs.PlateCarree())
# ax.set_xticks(np.arange(int(np.min(idx_lon)), int(np.max(idx_lon)), 1))
# ax.set_yticks(np.arange(int(np.min(idx_lat)), int(np.max(idx_lat)), 1))
# ax.xaxis.set_major_formatter(LongitudeFormatter())
# ax.yaxis.set_major_formatter(LatitudeFormatter())
# ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
# pcolor = plt.scatter(idx_lon, idx_lat, c = idx_bt[-1], transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18, s = 35, marker = 's')
# colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'm/s')
# plt.savefig(abspath + '/Save/Plot/Test/Check3.png', dpi = 1000, bbox_inches = 'tight')

# x = np.array([1, 4, 3, -1, 6, 9])
# print(x.argsort())

# a = [3, 1, 2, 2, 2, 5]
# a = np.array(a)
# # a = np.argsort(np.argsort(a))
# a = np.unique(a, return_counts = True)[1]
# print(a)

# print(random.sample(range(0, 6), 10))
# print(random.uniform(10, 20))

# list_a = [1, 2, 3]
# list_b = [9, 5, 4]
# list_a_b = [[a, b] for a, b in zip(list_a, list_b)]
# print(list_a_b)

# print(5 // 3)

# savename = 'GCN_{0}_{1}'.format(16, 1)
# file = open(abspath + '/Save/Pickle/Group1/{0}.pkl'.format(savename), 'rb')
# dic = pickle.load(file)
# file.close()
# print(dic['train_loss'])
# print(dic['val_output'])

# R = 6371
# lat_array = np.array([10, 10, 10])
# lon_array = np.array([180, -179.75, -179.5])
# distance_matrix = np.int64(R * np.arccos(np.sin(lat_array[:,None]/180*np.pi)*np.sin(lat_array[None,:]/180*np.pi) + \
#                                                   np.cos(lat_array[:,None]/180*np.pi)*np.cos(lat_array[None,:]/180*np.pi)*np.cos(lon_array[:,None]/180*np.pi-lon_array[None,:]/180*np.pi)))
# print(distance_matrix)

# a = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
# row, col = np.nonzero(a)
# values = a[row, col]
# ed = sp.coo_matrix((values, (row, col)))
# indices=np.vstack((ed.row,ed.col))
# index=torch.tensor(indices)
# value=torch.tensor(ed.data)
# edge_index=torch.sparse_coo_tensor(index,value,ed.shape)
# csr_a = csr_a.tocoo()
# csr_a = np.array(csr_a)
# print(ed.row)
# print(ed.col)
# print(csr_a.shape)

# print(np.arccos(-5))

# a = np.array([1, 2, 3, 3, 4])
# print(a[[0, 0]])

# loc = 'G:/DATA/Python/SWS/DataSet/X/FY3C'
# for orbit in ['asc', 'dsc']:
#     file_list = loc + '/{0}'.format(orbit.capitalize())
#     file_list = sorted([file_list + '/' + i for i in os.listdir(file_list)])
#     for file in file_list:
#         if file[-2:] == 'nc':
#             year = file.split('_')[-1][:4]
#             month = file.split('_')[-1][5:7]
#             day = file.split('_')[-1][8:10]
#             hour = file.split('_')[-1][11:13]
#             minute = file.split('_')[-1][14:15]
#             bt = xr.open_dataset(file)
#             bt_grid = np.array(bt['bt'])
#             lat_grid = np.array(bt['lat'])
#             lon_grid = np.array(bt['lon'])
#             saa = np.array(nc.Dataset('F:/FY3C/MWRI/{0}/{1}/FY3C_MWRIA_GBAL_L1_{2}{3}{4}_0120_010KM_MS.HDF'.format(year, orbit.capitalize(), year, month, day)).groups['Geolocation']['SensorAzimuth']).flatten()
#             print(np.max(saa))
#             print(np.min(saa))
#             lat_all = np.array(nc.Dataset('F:/FY3C/MWRI/{0}/{1}/FY3C_MWRIA_GBAL_L1_{2}{3}{4}_0120_010KM_MS.HDF'.format(year, orbit.capitalize(), year, month, day)).groups['Geolocation']['Latitude']).flatten()
#             lon_all = np.array(nc.Dataset('F:/FY3C/MWRI/{0}/{1}/FY3C_MWRIA_GBAL_L1_{2}{3}{4}_0120_010KM_MS.HDF'.format(year, orbit.capitalize(), year, month, day)).groups['Geolocation']['Longitude']).flatten()
#             points = np.concatenate([lon_all.reshape(-1, 1), lat_all.reshape(-1, 1)], axis = 1)
#             saa = griddata(points, saa.reshape(-1, 1), (lon_grid.reshape(-1, 1), lat_grid.reshape(-1, 1)), method = 'linear')[:, :, 0][:, 0]
#             print(saa)
#             print(np.max(saa))
#             print(np.min(saa))
#             saa = saa * 0.01 + 327.68
#             output = xr.Dataset({'bt':(['channel_num', 'bt_num'], bt_grid), 'lat':(['bt_num'], lat_grid), 'lon':(['bt_num'], lon_grid)}, coords = {'channel_num':np.arange(10), 'bt_num':np.arange(len(lat_grid))})

# a = np.arange(10)
# a = pd.DataFrame(a)
# a = a.shift(2)
# print(a)
# a = np.array(a)
# print(a)

# lon = np.arange(10, 20)
# lat = np.arange(0, 4)
# lon, lat = np.meshgrid(lon, lat)
# # print(lon)
# dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
# # print(dx)
# a = np.array([np.arange(20, 30) + 10, np.arange(50, 60), np.arange(70, 80), np.arange(100, 110)])
# # a[0][0] = np.nan
# # a = np.random.randn(4, 10)
# # print(a.shape)
# # a = mpcalc.laplacian(a, deltas = (dy, dx))
# a = mpcalc.gradient(a, deltas = (dy, dx))
# # print(a)
# # a = a.shift(2)
# # print(a)
# # a = np.array(a)
# a = np.array(a) * 111000
# print(a[0][0])
# # print(a[0].shape)

# loc_bt = abspath + '/DataSet/X/FY3'
# for orbit in ['asc', 'dsc']:
#     print(orbit)
#     bt_list = loc_bt + '/{0}'.format(orbit.capitalize())
#     bt_list = os.listdir(bt_list)
#     for bt in bt_list:
#         os.rename('/iapdisk2/Python/SWS/DataSet/X/FY3/{0}/{1}'.format(orbit.capitalize(), bt), '/iapdisk2/Python/SWS/DataSet/X/FY3/{0}/{1}'.format(orbit.capitalize(), bt.replace('FY3C', 'FY3')))

# data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# # print(data.shape)
# # print(data[[0, 1], [2, 3]])
# data = np.zeros([1441, 2881])
# # print(data.shape)
# # print(data[[0, 1], [2, 3]])
# print(data[[2, 5], [50, 2]])

# print(ord('a'))
# print(chr(97))

# result = os.popen('ps aux')  
# res = result.read()  
# for line in res.splitlines():  
#         print (line  )

# domain_list = [[0, 16, 24], [15, 31, 24], [30, 46, 32], [45, 61, 40], [-16, 0, 24], [-31, -15, 24], [-46, -30, 32], [-61, -45, 40], [60, 84, 24], [-84, -60, 24]]
# for domain in domain_list:
#     lat_min = domain[0]
#     lat_max = domain[1]
#     lon_len = domain[2]
#     file = open(abspath + '/Note/MeanStd/MeanStd_lat{0}to{1}_lon{2}.pkl'.format(lat_min, lat_max, lon_len), 'rb')
#     meanstd_dic = pickle.load(file)
#     file.close()
#     sswsst_mean_all = meanstd_dic['sswsst_mean']
#     sswsst_std_all = meanstd_dic['sswsst_std']
#     print('Mean_sswsst_lat{0}to{1}_lon{2}: {3}'.format(lat_min, lat_max, lon_len, sswsst_mean_all))
#     print('Std_sswsst_lat{0}to{1}_lon{2}: {3}'.format(lat_min, lat_max, lon_len, sswsst_std_all))

# print(np.mean([
# 0.89912,
# 0.87352,
# 0.94555,
# 0.78817,
# 0.81958,
# 0.80245,
# 0.89386,
# 0.89076,
# 0.87341,
# 0.91466,
# 0.81942,
# 0.84072,
# 0.79824
#  ]))

# print(torch.fmod(torch.tensor([-3]), 2.2))

# array1 = torch.tensor([1, 2, 3, 4, 5, 7])
# array2 = torch.tensor([3, 2, 4, 1, 6, 1])
# comparison = array1 > array2
# print(comparison)
# indices = torch.nonzero(comparison)
# print(indices)
# first_index = indices[0].item() if indices.numel() > 0 else -1
# print("第一个满足条件的索引:", first_index)

