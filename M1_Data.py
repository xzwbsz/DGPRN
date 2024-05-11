import os
import torch
import pickle
import random
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from torch.utils.data import Dataset

abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
loc_bt = abspath + '/DataSet/X/FY3'
loc_sswsst = abspath + '/DataSet/Y/ERA5'
asc_list = []
dsc_list = []
for orbit in ['asc', 'dsc']:
    bt_list = loc_bt + '/{0}'.format(orbit.capitalize())
    bt_list = sorted([bt_list + '/' + i for i in os.listdir(bt_list)])
    for file_bt in bt_list:
        if file_bt.split('_')[1] == orbit:
            if orbit == 'asc':
                asc_list.append([orbit, '{0}'.format(file_bt.split('_')[-1])])
            elif orbit == 'dsc':
                dsc_list.append([orbit, '{0}'.format(file_bt.split('_')[-1])])
trainval_list = asc_list + dsc_list
train_list = []
val_list = []
for data in trainval_list:
    if data[1].split('.')[0] in ['2022', '2023']:
        val_list.append(data)
    else:
        train_list.append(data)
# test_list = []
# test_date = ['2023-05-24', '2023-07-27', '2023-08-30']
# for data in trainval_list:
#     # if ('{0}-{1}-{2}'.format(data[1].split('.')[0], data[1].split('.')[1], data[1].split('.')[2]) in test_date):
#     if (data[1].split('.')[0] in ['2022']) and (data[0] == 'asc') and (data[1].split('.')[1] in ['01']):
#         test_list.append(data)
# test_list = test_list[int(len(test_list) / 2):]
class Data(Dataset):
    def __init__(self, datatype, lat_min, lat_max, lat_len, lon_len, device):
        super().__init__()
        if datatype == 'train':
            self.data_list = train_list
            # self.data_list = train_list[:5]
            # self.data_list = [val_list[67]]
        elif datatype == 'val':
            self.data_list = val_list[-500:]
            # self.data_list = val_list[:5]
            # self.data_list = [val_list[67]]
        self.timelen = len(self.data_list)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lat_len = lat_len
        self.lon_len = lon_len
        self.lat_grid = np.arange(lat_min, lat_max, 0.125)
        self.lon_grid = np.arange(0, lon_len, 0.125)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon_grid, self.lat_grid)
        self.grid_idx = np.arange(lat_len * 8 * lon_len * 8).reshape(lat_len * 8, lon_len * 8)
        self.device = torch.device(device)
        self.topography = torch.tensor(np.array(xr.open_dataset(abspath + '/DataSet/X/SHP0/SHP_topography.nc')['topography']), dtype = torch.float32).to(self.device)
        meanstd = open(abspath + '/Note/MeanStd1/MeanStd_lat{0}to{1}_lon{2}.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        meanstd_dic = pickle.load(meanstd)
        meanstd.close()
        self.bt_mean = np.array(meanstd_dic['bt_mean'])
        self.bt_std = np.array(meanstd_dic['bt_std'])
        self.sswsst_mean = np.array(meanstd_dic['sswsst_mean'])
        self.sswsst_std = np.array(meanstd_dic['sswsst_std'])
        distance_1 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_1.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_1 = pickle.load(distance_1)
        distance_1.close()
        self.distance_1 = np.array(distance_dic_1['distance'])
        distance_2 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_2.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_2 = pickle.load(distance_2)
        distance_2.close()
        self.distance_2 = np.array(distance_dic_2['distance'])
        distance_3 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_3.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_3 = pickle.load(distance_3)
        distance_3.close()
        self.distance_3 = np.array(distance_dic_3['distance'])
        distance_4 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_4.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_4 = pickle.load(distance_4)
        distance_4.close()
        self.distance_4 = np.array(distance_dic_4['distance'])
        distance_5 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_5.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_5 = pickle.load(distance_5)
        distance_5.close()
        self.distance_5 = np.array(distance_dic_5['distance'])
        linear = open(abspath + '/Note/LinearRegression/LinearRegression_lat{0}to{1}_lon{2}.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        linear_dic = pickle.load(linear)
        linear.close()
        self.linear = [np.array(linear_dic['a']).reshape(-1, 1), np.array(linear_dic['b'])]
        print('{0} length= {1}'.format(datatype, self.timelen))
    def __len__(self):
        return self.timelen
    def __getitem__(self, idx):
        slice_bt_domain_idx = []
        slice_bt_nan_idx = []
        slice_sswsst_domain_idx = []
        slice_sswsst_nan_idx = []
        slice_bt_domain_ramdom_idx = []
        slice_sswsst_domain_ramdom_idx = []
        slice_ice_idx = []
        while_num = 0
        min_num = 20
        while (len(slice_bt_domain_idx) <= min_num) or (len(slice_bt_nan_idx) <= min_num) or (len(slice_sswsst_domain_idx) <= min_num) or (len(slice_sswsst_nan_idx) <= min_num) or \
            (len(slice_bt_domain_ramdom_idx) <= min_num) or (len(slice_sswsst_domain_ramdom_idx) <= min_num) or (len(slice_ice_idx) <= min_num):
            if while_num > 0:
                idx = random.sample(range(self.timelen), 1)[0]
            while_num = while_num + 1
            idx_bt_data = xr.open_dataset(loc_bt + '/{0}/FY3_{1}_bt_{2}'.format(self.data_list[idx][0].capitalize(), self.data_list[idx][0], self.data_list[idx][1]))
            idx_sswsst_data = xr.open_dataset(loc_sswsst + '/{0}/ERA5_{1}_sswsst_{2}'.format(self.data_list[idx][0].capitalize(), self.data_list[idx][0], self.data_list[idx][1]))
            idx_bt = np.array(idx_bt_data['bt'])
            idx_bt_lat = np.array(idx_bt_data['lat'])
            idx_bt_lon = np.array(idx_bt_data['lon'])
            idx_saa = np.array(idx_bt_data['saa'])
            idx_sswsst = np.array(idx_sswsst_data['sswsst'])
            idx_sswsst_lat = np.array(idx_sswsst_data['lat'])
            idx_sswsst_lon = np.array(idx_sswsst_data['lon'])
            slice_bt_domain_idx = np.where((idx_bt_lat >= self.lat_min) & (idx_bt_lat < self.lat_max))[0]
            idx_bt = idx_bt[:, slice_bt_domain_idx]
            idx_bt_lat = idx_bt_lat[slice_bt_domain_idx]
            idx_bt_lon = idx_bt_lon[slice_bt_domain_idx]
            idx_saa = idx_saa[slice_bt_domain_idx]
            idx_sswsst = idx_sswsst[:, slice_bt_domain_idx]
            idx_sswsst_lat = idx_sswsst_lat[slice_bt_domain_idx]
            idx_sswsst_lon = idx_sswsst_lon[slice_bt_domain_idx]

            slice_bt_domain_ramdom_idx = np.zeros([min_num + 1])
            slice_sswsst_domain_ramdom_idx = np.zeros([min_num + 1])
            if len(idx_bt_lon) >= min_num:
                if (np.abs(self.lat_min) > 75) or (np.abs(self.lat_max) > 75):
                    ramdom_lon = idx_bt_lon[random.sample(range(len(idx_bt_lon)), 1)[0]]
                    ramdom_lon_start = ramdom_lon - 12
                    ramdom_lon_end = ramdom_lon + 12
                    if ramdom_lon_start < -180:
                        ramdom_lon_start = ramdom_lon_start + 360
                        slice_bt_domain_ramdom_idx = np.where((idx_bt_lon <= ramdom_lon_end) | (idx_bt_lon > ramdom_lon_start))[0]
                    elif ramdom_lon_end > 180:
                        ramdom_lon_end = ramdom_lon_end - 360
                        slice_bt_domain_ramdom_idx = np.where((idx_bt_lon >= ramdom_lon_start) | (idx_bt_lon < ramdom_lon_end))[0]
                    else:
                        slice_bt_domain_ramdom_idx = np.where((idx_bt_lon >= ramdom_lon_start) & (idx_bt_lon < ramdom_lon_end))[0]
                    idx_bt = idx_bt[:, slice_bt_domain_ramdom_idx]
                    idx_bt_lat = idx_bt_lat[slice_bt_domain_ramdom_idx]
                    idx_bt_lon = idx_bt_lon[slice_bt_domain_ramdom_idx]
                    idx_saa = idx_saa[slice_bt_domain_ramdom_idx]
                    idx_sswsst = idx_sswsst[:, slice_bt_domain_ramdom_idx]
                    idx_sswsst_lat = idx_sswsst_lat[slice_bt_domain_ramdom_idx]
                    idx_sswsst_lon = idx_sswsst_lon[slice_bt_domain_ramdom_idx]
                    ramdom_lon_start = ramdom_lon - 12
                    ramdom_lon_end = ramdom_lon + 12
                    if ramdom_lon_start < -180:
                        ramdom_lon_start = ramdom_lon_start + 360
                        slice_sswsst_domain_ramdom_idx = np.where((idx_sswsst_lon <= ramdom_lon_end) | (idx_sswsst_lon > ramdom_lon_start))[0]
                    elif ramdom_lon_end > 180:
                        ramdom_lon_end = ramdom_lon_end - 360
                        slice_sswsst_domain_ramdom_idx = np.where((idx_sswsst_lon >= ramdom_lon_start) | (idx_sswsst_lon < ramdom_lon_end))[0]
                    else:
                        slice_sswsst_domain_ramdom_idx = np.where((idx_sswsst_lon >= ramdom_lon_start) & (idx_sswsst_lon < ramdom_lon_end))[0]
                    idx_bt = idx_bt[:, slice_sswsst_domain_ramdom_idx]
                    idx_bt_lat = idx_bt_lat[slice_sswsst_domain_ramdom_idx]
                    idx_bt_lon = idx_bt_lon[slice_sswsst_domain_ramdom_idx]
                    idx_saa = idx_saa[slice_sswsst_domain_ramdom_idx]
                    idx_sswsst = idx_sswsst[:, slice_sswsst_domain_ramdom_idx]
                    idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_domain_ramdom_idx]
                    idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_domain_ramdom_idx]


            slice_bt_nan_idx = np.where((idx_bt <= 10) | (idx_bt >= 310))[1]
            slice_bt_nan_idx = np.setdiff1d(np.arange(idx_bt.shape[1]), slice_bt_nan_idx)
            idx_bt = idx_bt[:, slice_bt_nan_idx]
            idx_bt_lat = idx_bt_lat[slice_bt_nan_idx]
            idx_bt_lon = idx_bt_lon[slice_bt_nan_idx]
            idx_saa = idx_saa[slice_bt_nan_idx]
            idx_sswsst = idx_sswsst[:, slice_bt_nan_idx]
            idx_sswsst_lat = idx_sswsst_lat[slice_bt_nan_idx]
            idx_sswsst_lon = idx_sswsst_lon[slice_bt_nan_idx]
            slice_sswsst_domain_idx = np.where((idx_sswsst_lat >= self.lat_min) & (idx_sswsst_lat < self.lat_max))[0]
            idx_bt = idx_bt[:, slice_sswsst_domain_idx]
            idx_bt_lat = idx_bt_lat[slice_sswsst_domain_idx]
            idx_bt_lon = idx_bt_lon[slice_sswsst_domain_idx]
            idx_saa = idx_saa[slice_sswsst_domain_idx]
            idx_sswsst = idx_sswsst[:, slice_sswsst_domain_idx]
            idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_domain_idx]
            idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_domain_idx]

            slice_sswsst_nan_idx = np.where(np.isnan(idx_sswsst[2]) == False)[0]
            idx_bt = idx_bt[:, slice_sswsst_nan_idx]
            idx_bt_lat = idx_bt_lat[slice_sswsst_nan_idx]
            idx_bt_lon = idx_bt_lon[slice_sswsst_nan_idx]
            idx_saa = idx_saa[slice_sswsst_nan_idx]
            idx_sswsst = idx_sswsst[:, slice_sswsst_nan_idx]
            idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_nan_idx]
            idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_nan_idx]

            idx_sst = (np.dot(idx_bt.transpose(1, 0), self.linear[0]) + self.linear[1])[:, 0]
            slice_ice_idx = np.zeros([min_num + 1])
            if (np.abs(self.lat_min) > 45) or (np.abs(self.lat_max) > 45):
                slice_ice_idx = np.where(idx_sst >= 273.15)[0]
                idx_bt = idx_bt[:, slice_ice_idx]
                idx_bt_lat = idx_bt_lat[slice_ice_idx]
                idx_bt_lon = idx_bt_lon[slice_ice_idx]
                idx_saa = idx_saa[slice_ice_idx]
                idx_sst = idx_sst[slice_ice_idx]
                idx_sswsst = idx_sswsst[:, slice_ice_idx]
                idx_sswsst_lat = idx_sswsst_lat[slice_ice_idx]
                idx_sswsst_lon = idx_sswsst_lon[slice_ice_idx]


            # print(len(slice_bt_domain_idx), len(slice_bt_nan_idx), len(slice_sswsst_domain_idx), len(slice_sswsst_nan_idx), len(slice_bt_domain_ramdom_idx), len(slice_sswsst_domain_ramdom_idx), len(slice_ice_idx))

        idx_ssws = np.expand_dims(((idx_sswsst[0] ** 2) + (idx_sswsst[1] ** 2)) ** 0.5, axis = 0)
        idx_sswsst = np.concatenate([idx_ssws, idx_sswsst], axis = 0)

        for channel_num in range(10):
            idx_bt[channel_num] = (idx_bt[channel_num] - self.bt_mean[channel_num]) / (self.bt_std[channel_num])
        for var_num in range(4):
            idx_sswsst[var_num] = (idx_sswsst[var_num] - self.sswsst_mean[var_num]) / (self.sswsst_std[var_num])
        idx_bt = idx_bt.transpose(1, 0)
        idx_sswsst = idx_sswsst.transpose(1, 0)
        idx_bt = torch.tensor(idx_bt, dtype = torch.float32).to(self.device)
        idx_bt_lat = torch.tensor(idx_bt_lat, dtype = torch.float32).to(self.device)
        idx_bt_lon = torch.tensor(idx_bt_lon, dtype = torch.float32).to(self.device)
        idx_saa = torch.tensor(idx_saa, dtype = torch.float32).to(self.device)
        idx_sst = torch.tensor(idx_sst, dtype = torch.float32).to(self.device)
        idx_sswsst = torch.tensor(idx_sswsst, dtype = torch.float32).to(self.device)
        idx_sswsst_lat = torch.tensor(idx_sswsst_lat, dtype = torch.float32).to(self.device)
        idx_sswsst_lon = torch.tensor(idx_sswsst_lon, dtype = torch.float32).to(self.device)
        idx_topography = self.topography[(((-idx_bt_lat) + 90) * 8).type(torch.long), ((idx_bt_lon + 180) * 8).type(torch.long)]
        if ((torch.max(idx_bt_lon) - torch.min(idx_bt_lon)) > 180) or ((torch.max(idx_sswsst_lon) - torch.min(idx_sswsst_lon)) > 180):
            lon_adjust_idx = torch.where(idx_bt_lon < 0)
            idx_bt_lon[lon_adjust_idx] = idx_bt_lon[lon_adjust_idx] + 360
            lon_adjust_idx = torch.where(idx_sswsst_lon < 0)
            idx_sswsst_lon[lon_adjust_idx] = idx_sswsst_lon[lon_adjust_idx] + 360
        # idx_bt_lat = ((idx_bt_lat - torch.min(idx_bt_lat)) * 8).type(torch.long)
        idx_bt_lat = ((idx_bt_lat - self.lat_min) * 8).type(torch.long)
        idx_bt_lon = ((idx_bt_lon - torch.min(idx_bt_lon)) * 8).type(torch.long)
        # idx_sswsst_lat = ((idx_sswsst_lat - torch.min(idx_sswsst_lat)) * 4).type(torch.long)
        idx_sswsst_lat = ((idx_sswsst_lat - self.lat_min) * 4).type(torch.long)
        idx_sswsst_lon = ((idx_sswsst_lon - torch.min(idx_sswsst_lon)) * 4).type(torch.long)
        if (torch.max(idx_bt_lon) > (self.lon_len * 8)) or (torch.max(idx_sswsst_lon) > (self.lon_len * 4)):
            slice_bt_idx = torch.where(idx_bt_lon < (self.lon_len * 8))
            slice_sswsst_idx = torch.where(idx_sswsst_lon < (self.lon_len * 4))
            idx_bt_lat = idx_bt_lat[slice_bt_idx]
            idx_bt_lon = idx_bt_lon[slice_bt_idx]
            idx_bt = idx_bt[slice_bt_idx]
            idx_topography = idx_topography[slice_bt_idx]
            idx_saa = idx_saa[slice_bt_idx]
            idx_sst = idx_sst[slice_bt_idx]
            idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_idx]
            idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_idx]
            idx_sswsst = idx_sswsst[slice_sswsst_idx]
        sst_grid = np.full([self.lat_len * 8, self.lon_len * 8], np.nan)
        sst_grid[idx_bt_lat.cpu().numpy(), idx_bt_lon.cpu().numpy()] = idx_sst.cpu().numpy()
        dx, dy = mpcalc.lat_lon_grid_deltas(self.lon_grid, self.lat_grid)
        sst_gradient = np.array(mpcalc.gradient(sst_grid, deltas = (dy, dx)))
        sst_gradient = ((sst_gradient[0] ** 2 + sst_gradient[1] ** 2) ** 0.5) * 1000
        sst_gradient[0] = np.nan
        sst_gradient[-1] = np.nan
        distance = (-250) * sst_gradient + 40
        distance[np.where(distance < 15)] = 15
        distance_idx_2 = self.grid_idx[np.where(distance >= 13.875)]
        distance_idx_3 = self.grid_idx[np.where(distance >= (2 ** 0.5) * 13.875)]
        distance_idx_4 = self.grid_idx[np.where(distance >= 2 * 13.875)]
        distance_idx_5 = self.grid_idx[np.where(distance >= (5 ** 0.5) * 13.875)]
        adjacency_1 = np.array(self.distance_1)
        adjacency_2 = self.distance_2[:, np.where((np.in1d(self.distance_2[0], distance_idx_2) == True) & (np.in1d(self.distance_2[1], distance_idx_2) == True))[0]]
        adjacency_3 = self.distance_3[:, np.where((np.in1d(self.distance_3[0], distance_idx_3) == True) & (np.in1d(self.distance_3[1], distance_idx_3) == True))[0]]
        adjacency_4 = self.distance_4[:, np.where((np.in1d(self.distance_4[0], distance_idx_4) == True) & (np.in1d(self.distance_4[1], distance_idx_4) == True))[0]]
        adjacency_5 = self.distance_5[:, np.where((np.in1d(self.distance_5[0], distance_idx_5) == True) & (np.in1d(self.distance_5[1], distance_idx_5) == True))[0]]
        edge_idx_batch = np.concatenate([adjacency_1, adjacency_2, adjacency_3, adjacency_4, adjacency_5], axis = 1)
        edge_idx_start = edge_idx_batch[0]
        edge_idx_start = edge_idx_batch[:, np.where(np.isnan(sst_grid.flatten()[edge_idx_start]) == False)[0]]
        edge_idx_end = edge_idx_start[1]
        edge_idx_batch = edge_idx_start[:, np.where(np.isnan(sst_grid.flatten()[edge_idx_end]) == False)[0]]
        edge_idx_batch = torch.tensor(np.array(edge_idx_batch), dtype = torch.long).to(self.device)
        x = torch.zeros([self.lat_len * 8, self.lon_len * 8, 12]).to(self.device)
        x[idx_bt_lat, idx_bt_lon, :10] = idx_bt
        x[idx_bt_lat, idx_bt_lon, 10] = idx_topography
        x[idx_bt_lat, idx_bt_lon, 11] = idx_saa
        x = torch.flatten(x, start_dim = 0, end_dim = 1)
        edge_idx = torch.zeros(2, self.distance_1.shape[1] + self.distance_2.shape[1] + self.distance_3.shape[1] + self.distance_4.shape[1]).type(torch.long).to(self.device)
        edge_idx[:, :edge_idx_batch.shape[1]] = edge_idx_batch
        edge_idx[-1, -1] = edge_idx_batch.shape[1]
        y = torch.zeros([self.lat_len * 4, self.lon_len * 4, 4]).to(self.device)
        y[idx_sswsst_lat, idx_sswsst_lon] = idx_sswsst
        y = torch.flatten(y, start_dim = 0, end_dim = 1)
        mask_net = torch.zeros([self.lat_len * 8, self.lon_len * 8]).to(self.device)
        mask_net[idx_bt_lat, idx_bt_lon] = 1
        mask_net = torch.flatten(mask_net, start_dim = 0, end_dim = 1).type(torch.bool)
        mask_loss = torch.zeros([self.lat_len * 4, self.lon_len * 4, 4]).to(self.device)
        mask_loss[idx_sswsst_lat, idx_sswsst_lon] = 1
        mask_loss = torch.flatten(mask_loss, start_dim = 0, end_dim = 1).type(torch.bool)
        if (np.abs(self.lat_min) > 75) or (np.abs(self.lat_max) > 75):
            return x, edge_idx, y, mask_net, mask_loss, [self.data_list[idx][0] + '_' + str(ramdom_lon), self.data_list[idx][1]]
        else:
            return x, edge_idx, y, mask_net, mask_loss, self.data_list[idx]

class Concat(Dataset):
    def __init__(self, concat_list, lat_min, lat_max, lat_len, lon_len, device):
        super().__init__()
        # self.data_list = val_list[::20]
        # self.data_list = val_list[105:120]
        self.data_list = concat_list
        self.timelen = len(self.data_list)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lat_len = lat_len
        self.lon_len = lon_len
        self.lat_grid = np.arange(lat_min, lat_max, 0.125)
        self.lon_grid = np.arange(0, lon_len, 0.125)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon_grid, self.lat_grid)
        self.grid_idx = np.arange(lat_len * 8 * lon_len * 8).reshape(lat_len * 8, lon_len * 8)
        self.device = torch.device(device)
        self.topography = torch.tensor(np.array(xr.open_dataset(abspath + '/DataSet/X/SHP0/SHP_topography.nc')['topography']), dtype = torch.float32).to(self.device)
        meanstd = open(abspath + '/Note/MeanStd1/MeanStd_lat{0}to{1}_lon{2}.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        meanstd_dic = pickle.load(meanstd)
        meanstd.close()
        self.bt_mean = np.array(meanstd_dic['bt_mean'])
        self.bt_std = np.array(meanstd_dic['bt_std'])
        self.sswsst_mean = np.array(meanstd_dic['sswsst_mean'])
        self.sswsst_std = np.array(meanstd_dic['sswsst_std'])
        distance_1 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_1.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_1 = pickle.load(distance_1)
        distance_1.close()
        self.distance_1 = np.array(distance_dic_1['distance'])
        distance_2 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_2.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_2 = pickle.load(distance_2)
        distance_2.close()
        self.distance_2 = np.array(distance_dic_2['distance'])
        distance_3 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_3.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_3 = pickle.load(distance_3)
        distance_3.close()
        self.distance_3 = np.array(distance_dic_3['distance'])
        distance_4 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_4.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_4 = pickle.load(distance_4)
        distance_4.close()
        self.distance_4 = np.array(distance_dic_4['distance'])
        distance_5 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_5.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        distance_dic_5 = pickle.load(distance_5)
        distance_5.close()
        self.distance_5 = np.array(distance_dic_5['distance'])
        linear = open(abspath + '/Note/LinearRegression/LinearRegression_lat{0}to{1}_lon{2}.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        linear_dic = pickle.load(linear)
        linear.close()
        self.linear = [np.array(linear_dic['a']).reshape(-1, 1), np.array(linear_dic['b'])]
        print('lat{0}to{1}_lon{2}, cancat length= {3}'.format(lat_min, lat_max, lon_len, self.timelen))
    def __len__(self):
        return self.timelen
    def __getitem__(self, idx):
        slice_bt_domain_idx = []
        slice_bt_nan_idx = []
        slice_sswsst_domain_idx = []
        slice_sswsst_nan_idx = []
        slice_bt_domain_ramdom_idx = []
        slice_sswsst_domain_ramdom_idx = []
        slice_ice_idx = []
        min_num = 20
        idx_bt_data = xr.open_dataset(loc_bt + '/{0}/FY3_{1}_bt_{2}'.format(self.data_list[idx][0].capitalize(), self.data_list[idx][0], self.data_list[idx][1]))
        idx_sswsst_data = xr.open_dataset(loc_sswsst + '/{0}/ERA5_{1}_sswsst_{2}'.format(self.data_list[idx][0].capitalize(), self.data_list[idx][0], self.data_list[idx][1]))
        idx_bt = np.array(idx_bt_data['bt'])
        idx_bt_lat = np.array(idx_bt_data['lat'])
        idx_bt_lon = np.array(idx_bt_data['lon'])
        idx_saa = np.array(idx_bt_data['saa'])
        idx_sswsst = np.array(idx_sswsst_data['sswsst'])
        idx_sswsst_lat = np.array(idx_sswsst_data['lat'])
        idx_sswsst_lon = np.array(idx_sswsst_data['lon'])
        slice_bt_domain_idx = np.where((idx_bt_lat >= self.lat_min) & (idx_bt_lat < self.lat_max))[0]
        idx_bt = idx_bt[:, slice_bt_domain_idx]
        idx_bt_lat = idx_bt_lat[slice_bt_domain_idx]
        idx_bt_lon = idx_bt_lon[slice_bt_domain_idx]
        idx_saa = idx_saa[slice_bt_domain_idx]
        idx_sswsst = idx_sswsst[:, slice_bt_domain_idx]
        idx_sswsst_lat = idx_sswsst_lat[slice_bt_domain_idx]
        idx_sswsst_lon = idx_sswsst_lon[slice_bt_domain_idx]
        slice_bt_domain_ramdom_idx = np.zeros([min_num + 1])
        slice_sswsst_domain_ramdom_idx = np.zeros([min_num + 1])




        if (np.abs(self.lat_min) > 75) or (np.abs(self.lat_max) > 75):
            x_all = []
            edge_idx_all = []
            y_all = []
            mask_net_all = []
            mask_loss_all = []
            lon_start_all = []

            idx_bt_all = idx_bt.copy()
            idx_bt_lat_all = idx_bt_lat.copy()
            idx_bt_lon_all = idx_bt_lon.copy()
            idx_saa_all = idx_saa.copy()
            idx_sswsst_all = idx_sswsst.copy()
            idx_sswsst_lat_all = idx_sswsst_lat.copy()
            idx_sswsst_lon_all = idx_sswsst_lon.copy()

            slice_bt_all_domain_ramdom_idx = np.zeros([min_num + 1])
            slice_sswsst_all_domain_ramdom_idx = np.zeros([min_num + 1])
            while (len(slice_bt_all_domain_ramdom_idx) > min_num) and (len(slice_sswsst_all_domain_ramdom_idx) > min_num) and (len(idx_bt_lon_all) > min_num):
                ramdom_lon_start = np.min(idx_bt_lon_all)
                ramdom_lon_end = ramdom_lon_start + 24
                slice_bt_domain_ramdom_idx = np.where((idx_bt_lon_all >= ramdom_lon_start) & (idx_bt_lon_all < ramdom_lon_end))[0]
                idx_bt = idx_bt_all[:, slice_bt_domain_ramdom_idx]
                idx_bt_lat = idx_bt_lat_all[slice_bt_domain_ramdom_idx]
                idx_bt_lon = idx_bt_lon_all[slice_bt_domain_ramdom_idx]
                idx_saa = idx_saa_all[slice_bt_domain_ramdom_idx]
                idx_sswsst = idx_sswsst_all[:, slice_bt_domain_ramdom_idx]
                idx_sswsst_lat = idx_sswsst_lat_all[slice_bt_domain_ramdom_idx]
                idx_sswsst_lon = idx_sswsst_lon_all[slice_bt_domain_ramdom_idx]
                slice_sswsst_domain_ramdom_idx = np.where((idx_sswsst_lon >= ramdom_lon_start) & (idx_sswsst_lon < ramdom_lon_end))[0]
                idx_bt = idx_bt[:, slice_sswsst_domain_ramdom_idx]
                idx_bt_lat = idx_bt_lat[slice_sswsst_domain_ramdom_idx]
                idx_bt_lon = idx_bt_lon[slice_sswsst_domain_ramdom_idx]
                idx_saa = idx_saa[slice_sswsst_domain_ramdom_idx]
                idx_sswsst = idx_sswsst[:, slice_sswsst_domain_ramdom_idx]
                idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_domain_ramdom_idx]
                idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_domain_ramdom_idx]
                slice_bt_all_domain_ramdom_idx = np.where((idx_bt_lon_all >= ramdom_lon_end))[0]
                idx_bt_all = idx_bt_all[:, slice_bt_all_domain_ramdom_idx]
                idx_bt_lat_all = idx_bt_lat_all[slice_bt_all_domain_ramdom_idx]
                idx_bt_lon_all = idx_bt_lon_all[slice_bt_all_domain_ramdom_idx]
                idx_saa_all = idx_saa_all[slice_bt_all_domain_ramdom_idx]
                idx_sswsst_all = idx_sswsst_all[:, slice_bt_all_domain_ramdom_idx]
                idx_sswsst_lat_all = idx_sswsst_lat_all[slice_bt_all_domain_ramdom_idx]
                idx_sswsst_lon_all = idx_sswsst_lon_all[slice_bt_all_domain_ramdom_idx]
                slice_sswsst_all_domain_ramdom_idx = np.where((idx_sswsst_lon_all >= ramdom_lon_end))[0]
                idx_bt_all = idx_bt_all[:, slice_sswsst_all_domain_ramdom_idx]
                idx_bt_lat_all = idx_bt_lat_all[slice_sswsst_all_domain_ramdom_idx]
                idx_bt_lon_all = idx_bt_lon_all[slice_sswsst_all_domain_ramdom_idx]
                idx_saa_all = idx_saa_all[slice_sswsst_all_domain_ramdom_idx]
                idx_sswsst_all = idx_sswsst_all[:, slice_sswsst_all_domain_ramdom_idx]
                idx_sswsst_lat_all = idx_sswsst_lat_all[slice_sswsst_all_domain_ramdom_idx]
                idx_sswsst_lon_all = idx_sswsst_lon_all[slice_sswsst_all_domain_ramdom_idx]

                slice_bt_nan_idx = np.where((idx_bt <= 10) | (idx_bt >= 310))[1]
                slice_bt_nan_idx = np.setdiff1d(np.arange(idx_bt.shape[1]), slice_bt_nan_idx)
                idx_bt = idx_bt[:, slice_bt_nan_idx]
                idx_bt_lat = idx_bt_lat[slice_bt_nan_idx]
                idx_bt_lon = idx_bt_lon[slice_bt_nan_idx]
                idx_saa = idx_saa[slice_bt_nan_idx]
                idx_sswsst = idx_sswsst[:, slice_bt_nan_idx]
                idx_sswsst_lat = idx_sswsst_lat[slice_bt_nan_idx]
                idx_sswsst_lon = idx_sswsst_lon[slice_bt_nan_idx]
                slice_sswsst_domain_idx = np.where((idx_sswsst_lat >= self.lat_min) & (idx_sswsst_lat < self.lat_max))[0]
                idx_bt = idx_bt[:, slice_sswsst_domain_idx]
                idx_bt_lat = idx_bt_lat[slice_sswsst_domain_idx]
                idx_bt_lon = idx_bt_lon[slice_sswsst_domain_idx]
                idx_saa = idx_saa[slice_sswsst_domain_idx]
                idx_sswsst = idx_sswsst[:, slice_sswsst_domain_idx]
                idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_domain_idx]
                idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_domain_idx]
                slice_sswsst_nan_idx = np.where(np.isnan(idx_sswsst[2]) == False)[0]
                idx_bt = idx_bt[:, slice_sswsst_nan_idx]
                idx_bt_lat = idx_bt_lat[slice_sswsst_nan_idx]
                idx_bt_lon = idx_bt_lon[slice_sswsst_nan_idx]
                idx_saa = idx_saa[slice_sswsst_nan_idx]
                idx_sswsst = idx_sswsst[:, slice_sswsst_nan_idx]
                idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_nan_idx]
                idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_nan_idx]
                idx_sst = (np.dot(idx_bt.transpose(1, 0), self.linear[0]) + self.linear[1])[:, 0]
                slice_ice_idx = np.zeros([min_num + 1])
                if (np.abs(self.lat_min) > 45) or (np.abs(self.lat_max) > 45):
                    slice_ice_idx = np.where(idx_sst >= 273.15)[0]
                    idx_bt = idx_bt[:, slice_ice_idx]
                    idx_bt_lat = idx_bt_lat[slice_ice_idx]
                    idx_bt_lon = idx_bt_lon[slice_ice_idx]
                    idx_saa = idx_saa[slice_ice_idx]
                    idx_sst = idx_sst[slice_ice_idx]
                    idx_sswsst = idx_sswsst[:, slice_ice_idx]
                    idx_sswsst_lat = idx_sswsst_lat[slice_ice_idx]
                    idx_sswsst_lon = idx_sswsst_lon[slice_ice_idx]
                if (len(slice_bt_domain_idx) <= min_num) or (len(slice_bt_nan_idx) <= min_num) or (len(slice_sswsst_domain_idx) <= min_num) or (len(slice_sswsst_nan_idx) <= min_num) or \
                    (len(slice_bt_domain_ramdom_idx) <= min_num) or (len(slice_sswsst_domain_ramdom_idx) <= min_num) or (len(slice_ice_idx) <= min_num):
                    pass
                else:
                    idx_ssws = np.expand_dims(((idx_sswsst[0] ** 2) + (idx_sswsst[1] ** 2)) ** 0.5, axis = 0)
                    idx_sswsst = np.concatenate([idx_ssws, idx_sswsst], axis = 0)
                    for channel_num in range(10):
                        idx_bt[channel_num] = (idx_bt[channel_num] - self.bt_mean[channel_num]) / (self.bt_std[channel_num])
                    for var_num in range(4):
                        idx_sswsst[var_num] = (idx_sswsst[var_num] - self.sswsst_mean[var_num]) / (self.sswsst_std[var_num])
                    idx_bt = idx_bt.transpose(1, 0)
                    idx_sswsst = idx_sswsst.transpose(1, 0)
                    idx_bt = torch.tensor(idx_bt, dtype = torch.float32).to(self.device)
                    idx_bt_lat = torch.tensor(idx_bt_lat, dtype = torch.float32).to(self.device)
                    idx_bt_lon = torch.tensor(idx_bt_lon, dtype = torch.float32).to(self.device)
                    idx_saa = torch.tensor(idx_saa, dtype = torch.float32).to(self.device)
                    idx_sst = torch.tensor(idx_sst, dtype = torch.float32).to(self.device)
                    idx_sswsst = torch.tensor(idx_sswsst, dtype = torch.float32).to(self.device)
                    idx_sswsst_lat = torch.tensor(idx_sswsst_lat, dtype = torch.float32).to(self.device)
                    idx_sswsst_lon = torch.tensor(idx_sswsst_lon, dtype = torch.float32).to(self.device)
                    idx_topography = self.topography[(((-idx_bt_lat) + 90) * 8).type(torch.long), ((idx_bt_lon + 180) * 8).type(torch.long)]
                    if ((torch.max(idx_bt_lon) - torch.min(idx_bt_lon)) > 180) or ((torch.max(idx_sswsst_lon) - torch.min(idx_sswsst_lon)) > 180):
                        lon_adjust_idx = torch.where(idx_bt_lon < 0)
                        idx_bt_lon[lon_adjust_idx] = idx_bt_lon[lon_adjust_idx] + 360
                        lon_adjust_idx = torch.where(idx_sswsst_lon < 0)
                        idx_sswsst_lon[lon_adjust_idx] = idx_sswsst_lon[lon_adjust_idx] + 360
                    lon_start = torch.min(idx_sswsst_lon) * 4
                    idx_bt_lat = ((idx_bt_lat - self.lat_min) * 8).type(torch.long)
                    idx_bt_lon = ((idx_bt_lon - torch.min(idx_bt_lon)) * 8).type(torch.long)
                    idx_sswsst_lat = ((idx_sswsst_lat - self.lat_min) * 4).type(torch.long)
                    idx_sswsst_lon = ((idx_sswsst_lon - torch.min(idx_sswsst_lon)) * 4).type(torch.long)
                    if (torch.max(idx_bt_lon) > (self.lon_len * 8)) or (torch.max(idx_sswsst_lon) > (self.lon_len * 4)):
                        slice_bt_idx = torch.where(idx_bt_lon < (self.lon_len * 8))
                        slice_sswsst_idx = torch.where(idx_sswsst_lon < (self.lon_len * 4))
                        idx_bt_lat = idx_bt_lat[slice_bt_idx]
                        idx_bt_lon = idx_bt_lon[slice_bt_idx]
                        idx_bt = idx_bt[slice_bt_idx]
                        idx_topography = idx_topography[slice_bt_idx]
                        idx_saa = idx_saa[slice_bt_idx]
                        idx_sst = idx_sst[slice_bt_idx]
                        idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_idx]
                        idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_idx]
                        idx_sswsst = idx_sswsst[slice_sswsst_idx]
                    sst_grid = np.full([self.lat_len * 8, self.lon_len * 8], np.nan)
                    sst_grid[idx_bt_lat.cpu().numpy(), idx_bt_lon.cpu().numpy()] = idx_sst.cpu().numpy()
                    dx, dy = mpcalc.lat_lon_grid_deltas(self.lon_grid, self.lat_grid)
                    sst_gradient = np.array(mpcalc.gradient(sst_grid, deltas = (dy, dx)))
                    sst_gradient = ((sst_gradient[0] ** 2 + sst_gradient[1] ** 2) ** 0.5) * 1000
                    sst_gradient[0] = np.nan
                    sst_gradient[-1] = np.nan
                    distance = (-250) * sst_gradient + 40
                    distance[np.where(distance < 15)] = 15
                    distance_idx_2 = self.grid_idx[np.where(distance >= 13.875)]
                    distance_idx_3 = self.grid_idx[np.where(distance >= (2 ** 0.5) * 13.875)]
                    distance_idx_4 = self.grid_idx[np.where(distance >= 2 * 13.875)]
                    distance_idx_5 = self.grid_idx[np.where(distance >= (5 ** 0.5) * 13.875)]
                    adjacency_1 = np.array(self.distance_1)
                    adjacency_2 = self.distance_2[:, np.where((np.in1d(self.distance_2[0], distance_idx_2) == True) & (np.in1d(self.distance_2[1], distance_idx_2) == True))[0]]
                    adjacency_3 = self.distance_3[:, np.where((np.in1d(self.distance_3[0], distance_idx_3) == True) & (np.in1d(self.distance_3[1], distance_idx_3) == True))[0]]
                    adjacency_4 = self.distance_4[:, np.where((np.in1d(self.distance_4[0], distance_idx_4) == True) & (np.in1d(self.distance_4[1], distance_idx_4) == True))[0]]
                    adjacency_5 = self.distance_5[:, np.where((np.in1d(self.distance_5[0], distance_idx_5) == True) & (np.in1d(self.distance_5[1], distance_idx_5) == True))[0]]
                    edge_idx_batch = np.concatenate([adjacency_1, adjacency_2, adjacency_3, adjacency_4, adjacency_5], axis = 1)
                    edge_idx_start = edge_idx_batch[0]
                    edge_idx_start = edge_idx_batch[:, np.where(np.isnan(sst_grid.flatten()[edge_idx_start]) == False)[0]]
                    edge_idx_end = edge_idx_start[1]
                    edge_idx_batch = edge_idx_start[:, np.where(np.isnan(sst_grid.flatten()[edge_idx_end]) == False)[0]]
                    edge_idx_batch = torch.tensor(np.array(edge_idx_batch), dtype = torch.long).to(self.device)
                    x = torch.zeros([self.lat_len * 8, self.lon_len * 8, 12]).to(self.device)
                    x[idx_bt_lat, idx_bt_lon, :10] = idx_bt
                    x[idx_bt_lat, idx_bt_lon, 10] = idx_topography
                    x[idx_bt_lat, idx_bt_lon, 11] = idx_saa
                    x = torch.flatten(x, start_dim = 0, end_dim = 1)
                    edge_idx = torch.zeros(2, self.distance_1.shape[1] + self.distance_2.shape[1] + self.distance_3.shape[1] + self.distance_4.shape[1]).type(torch.long).to(self.device)
                    edge_idx[:, :edge_idx_batch.shape[1]] = edge_idx_batch
                    edge_idx[-1, -1] = edge_idx_batch.shape[1]
                    y = torch.zeros([self.lat_len * 4, self.lon_len * 4, 4]).to(self.device)
                    y[idx_sswsst_lat, idx_sswsst_lon] = idx_sswsst
                    y = torch.flatten(y, start_dim = 0, end_dim = 1)
                    mask_net = torch.zeros([self.lat_len * 8, self.lon_len * 8]).to(self.device)
                    mask_net[idx_bt_lat, idx_bt_lon] = 1
                    mask_net = torch.flatten(mask_net, start_dim = 0, end_dim = 1).type(torch.bool)
                    mask_loss = torch.zeros([self.lat_len * 4, self.lon_len * 4, 4]).to(self.device)
                    mask_loss[idx_sswsst_lat, idx_sswsst_lon] = 1
                    mask_loss = torch.flatten(mask_loss, start_dim = 0, end_dim = 1).type(torch.bool)
                    x_all.append(x)
                    edge_idx_all.append(edge_idx)
                    y_all.append(y)
                    mask_net_all.append(mask_net)
                    mask_loss_all.append(mask_loss)
                    lon_start_all.append(lon_start)
            if len(x_all) == 0:
                return self.data_list[idx]
            else:
                x_all = torch.stack(x_all)
                edge_idx_all = torch.stack(edge_idx_all)
                y_all = torch.stack(y_all)
                mask_net_all = torch.stack(mask_net_all)
                mask_loss_all = torch.stack(mask_loss_all)
                lon_start_all = torch.stack(lon_start_all)
                return x_all, edge_idx_all, y_all, mask_net_all, mask_loss_all, self.data_list[idx], lon_start_all



        else:
            slice_bt_nan_idx = np.where((idx_bt <= 10) | (idx_bt >= 310))[1]
            slice_bt_nan_idx = np.setdiff1d(np.arange(idx_bt.shape[1]), slice_bt_nan_idx)
            idx_bt = idx_bt[:, slice_bt_nan_idx]
            idx_bt_lat = idx_bt_lat[slice_bt_nan_idx]
            idx_bt_lon = idx_bt_lon[slice_bt_nan_idx]
            idx_saa = idx_saa[slice_bt_nan_idx]
            idx_sswsst = idx_sswsst[:, slice_bt_nan_idx]
            idx_sswsst_lat = idx_sswsst_lat[slice_bt_nan_idx]
            idx_sswsst_lon = idx_sswsst_lon[slice_bt_nan_idx]
            slice_sswsst_domain_idx = np.where((idx_sswsst_lat >= self.lat_min) & (idx_sswsst_lat < self.lat_max))[0]
            idx_bt = idx_bt[:, slice_sswsst_domain_idx]
            idx_bt_lat = idx_bt_lat[slice_sswsst_domain_idx]
            idx_bt_lon = idx_bt_lon[slice_sswsst_domain_idx]
            idx_saa = idx_saa[slice_sswsst_domain_idx]
            idx_sswsst = idx_sswsst[:, slice_sswsst_domain_idx]
            idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_domain_idx]
            idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_domain_idx]
            slice_sswsst_nan_idx = np.where(np.isnan(idx_sswsst[2]) == False)[0]
            idx_bt = idx_bt[:, slice_sswsst_nan_idx]
            idx_bt_lat = idx_bt_lat[slice_sswsst_nan_idx]
            idx_bt_lon = idx_bt_lon[slice_sswsst_nan_idx]
            idx_saa = idx_saa[slice_sswsst_nan_idx]
            idx_sswsst = idx_sswsst[:, slice_sswsst_nan_idx]
            idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_nan_idx]
            idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_nan_idx]
            idx_sst = (np.dot(idx_bt.transpose(1, 0), self.linear[0]) + self.linear[1])[:, 0]
            slice_ice_idx = np.zeros([min_num + 1])
            if (np.abs(self.lat_min) > 45) or (np.abs(self.lat_max) > 45):
                slice_ice_idx = np.where(idx_sst >= 273.15)[0]
                idx_bt = idx_bt[:, slice_ice_idx]
                idx_bt_lat = idx_bt_lat[slice_ice_idx]
                idx_bt_lon = idx_bt_lon[slice_ice_idx]
                idx_saa = idx_saa[slice_ice_idx]
                idx_sst = idx_sst[slice_ice_idx]
                idx_sswsst = idx_sswsst[:, slice_ice_idx]
                idx_sswsst_lat = idx_sswsst_lat[slice_ice_idx]
                idx_sswsst_lon = idx_sswsst_lon[slice_ice_idx]
            if (len(slice_bt_domain_idx) <= min_num) or (len(slice_bt_nan_idx) <= min_num) or (len(slice_sswsst_domain_idx) <= min_num) or (len(slice_sswsst_nan_idx) <= min_num) or \
                (len(slice_bt_domain_ramdom_idx) <= min_num) or (len(slice_sswsst_domain_ramdom_idx) <= min_num) or (len(slice_ice_idx) <= min_num):
                return self.data_list[idx]
            else:
                idx_ssws = np.expand_dims(((idx_sswsst[0] ** 2) + (idx_sswsst[1] ** 2)) ** 0.5, axis = 0)
                idx_sswsst = np.concatenate([idx_ssws, idx_sswsst], axis = 0)
                for channel_num in range(10):
                    idx_bt[channel_num] = (idx_bt[channel_num] - self.bt_mean[channel_num]) / (self.bt_std[channel_num])
                for var_num in range(4):
                    idx_sswsst[var_num] = (idx_sswsst[var_num] - self.sswsst_mean[var_num]) / (self.sswsst_std[var_num])
                idx_bt = idx_bt.transpose(1, 0)
                idx_sswsst = idx_sswsst.transpose(1, 0)
                idx_bt = torch.tensor(idx_bt, dtype = torch.float32).to(self.device)
                idx_bt_lat = torch.tensor(idx_bt_lat, dtype = torch.float32).to(self.device)
                idx_bt_lon = torch.tensor(idx_bt_lon, dtype = torch.float32).to(self.device)
                idx_saa = torch.tensor(idx_saa, dtype = torch.float32).to(self.device)
                idx_sst = torch.tensor(idx_sst, dtype = torch.float32).to(self.device)
                idx_sswsst = torch.tensor(idx_sswsst, dtype = torch.float32).to(self.device)
                idx_sswsst_lat = torch.tensor(idx_sswsst_lat, dtype = torch.float32).to(self.device)
                idx_sswsst_lon = torch.tensor(idx_sswsst_lon, dtype = torch.float32).to(self.device)
                idx_topography = self.topography[(((-idx_bt_lat) + 90) * 8).type(torch.long), ((idx_bt_lon + 180) * 8).type(torch.long)]
                if ((torch.max(idx_bt_lon) - torch.min(idx_bt_lon)) > 180) or ((torch.max(idx_sswsst_lon) - torch.min(idx_sswsst_lon)) > 180):
                    lon_adjust_idx = torch.where(idx_bt_lon < 0)
                    idx_bt_lon[lon_adjust_idx] = idx_bt_lon[lon_adjust_idx] + 360
                    lon_adjust_idx = torch.where(idx_sswsst_lon < 0)
                    idx_sswsst_lon[lon_adjust_idx] = idx_sswsst_lon[lon_adjust_idx] + 360
                lon_start = torch.min(idx_sswsst_lon) * 4
                idx_bt_lat = ((idx_bt_lat - self.lat_min) * 8).type(torch.long)
                idx_bt_lon = ((idx_bt_lon - torch.min(idx_bt_lon)) * 8).type(torch.long)
                idx_sswsst_lat = ((idx_sswsst_lat - self.lat_min) * 4).type(torch.long)
                idx_sswsst_lon = ((idx_sswsst_lon - torch.min(idx_sswsst_lon)) * 4).type(torch.long)
                if (torch.max(idx_bt_lon) > (self.lon_len * 8)) or (torch.max(idx_sswsst_lon) > (self.lon_len * 4)):
                    slice_bt_idx = torch.where(idx_bt_lon < (self.lon_len * 8))
                    slice_sswsst_idx = torch.where(idx_sswsst_lon < (self.lon_len * 4))
                    idx_bt_lat = idx_bt_lat[slice_bt_idx]
                    idx_bt_lon = idx_bt_lon[slice_bt_idx]
                    idx_bt = idx_bt[slice_bt_idx]
                    idx_topography = idx_topography[slice_bt_idx]
                    idx_saa = idx_saa[slice_bt_idx]
                    idx_sst = idx_sst[slice_bt_idx]
                    idx_sswsst_lat = idx_sswsst_lat[slice_sswsst_idx]
                    idx_sswsst_lon = idx_sswsst_lon[slice_sswsst_idx]
                    idx_sswsst = idx_sswsst[slice_sswsst_idx]
                sst_grid = np.full([self.lat_len * 8, self.lon_len * 8], np.nan)
                sst_grid[idx_bt_lat.cpu().numpy(), idx_bt_lon.cpu().numpy()] = idx_sst.cpu().numpy()
                dx, dy = mpcalc.lat_lon_grid_deltas(self.lon_grid, self.lat_grid)
                sst_gradient = np.array(mpcalc.gradient(sst_grid, deltas = (dy, dx)))
                sst_gradient = ((sst_gradient[0] ** 2 + sst_gradient[1] ** 2) ** 0.5) * 1000
                sst_gradient[0] = np.nan
                sst_gradient[-1] = np.nan
                distance = (-250) * sst_gradient + 40
                distance[np.where(distance < 15)] = 15
                distance_idx_2 = self.grid_idx[np.where(distance >= 13.875)]
                distance_idx_3 = self.grid_idx[np.where(distance >= (2 ** 0.5) * 13.875)]
                distance_idx_4 = self.grid_idx[np.where(distance >= 2 * 13.875)]
                distance_idx_5 = self.grid_idx[np.where(distance >= (5 ** 0.5) * 13.875)]
                adjacency_1 = np.array(self.distance_1)
                adjacency_2 = self.distance_2[:, np.where((np.in1d(self.distance_2[0], distance_idx_2) == True) & (np.in1d(self.distance_2[1], distance_idx_2) == True))[0]]
                adjacency_3 = self.distance_3[:, np.where((np.in1d(self.distance_3[0], distance_idx_3) == True) & (np.in1d(self.distance_3[1], distance_idx_3) == True))[0]]
                adjacency_4 = self.distance_4[:, np.where((np.in1d(self.distance_4[0], distance_idx_4) == True) & (np.in1d(self.distance_4[1], distance_idx_4) == True))[0]]
                adjacency_5 = self.distance_5[:, np.where((np.in1d(self.distance_5[0], distance_idx_5) == True) & (np.in1d(self.distance_5[1], distance_idx_5) == True))[0]]
                edge_idx_batch = np.concatenate([adjacency_1, adjacency_2, adjacency_3, adjacency_4, adjacency_5], axis = 1)
                edge_idx_start = edge_idx_batch[0]
                edge_idx_start = edge_idx_batch[:, np.where(np.isnan(sst_grid.flatten()[edge_idx_start]) == False)[0]]
                edge_idx_end = edge_idx_start[1]
                edge_idx_batch = edge_idx_start[:, np.where(np.isnan(sst_grid.flatten()[edge_idx_end]) == False)[0]]
                edge_idx_batch = torch.tensor(np.array(edge_idx_batch), dtype = torch.long).to(self.device)
                x = torch.zeros([self.lat_len * 8, self.lon_len * 8, 12]).to(self.device)
                x[idx_bt_lat, idx_bt_lon, :10] = idx_bt
                x[idx_bt_lat, idx_bt_lon, 10] = idx_topography
                x[idx_bt_lat, idx_bt_lon, 11] = idx_saa
                x = torch.flatten(x, start_dim = 0, end_dim = 1)
                edge_idx = torch.zeros(2, self.distance_1.shape[1] + self.distance_2.shape[1] + self.distance_3.shape[1] + self.distance_4.shape[1]).type(torch.long).to(self.device)
                edge_idx[:, :edge_idx_batch.shape[1]] = edge_idx_batch
                edge_idx[-1, -1] = edge_idx_batch.shape[1]
                y = torch.zeros([self.lat_len * 4, self.lon_len * 4, 4]).to(self.device)
                y[idx_sswsst_lat, idx_sswsst_lon] = idx_sswsst
                y = torch.flatten(y, start_dim = 0, end_dim = 1)
                mask_net = torch.zeros([self.lat_len * 8, self.lon_len * 8]).to(self.device)
                mask_net[idx_bt_lat, idx_bt_lon] = 1
                mask_net = torch.flatten(mask_net, start_dim = 0, end_dim = 1).type(torch.bool)
                mask_loss = torch.zeros([self.lat_len * 4, self.lon_len * 4, 4]).to(self.device)
                mask_loss[idx_sswsst_lat, idx_sswsst_lon] = 1
                mask_loss = torch.flatten(mask_loss, start_dim = 0, end_dim = 1).type(torch.bool)
                return x, edge_idx, y, mask_net, mask_loss, self.data_list[idx], lon_start