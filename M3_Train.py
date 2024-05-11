import os
import time
import torch
import pickle
import random
import datetime
import numpy as np
import xarray as xr
import pandas as pd
from itertools import groupby
from collections import OrderedDict
from torch.utils.data import DataLoader

abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def train(M1_Data, M2_DGPRN, group, num, domain, epoch_num, batch_size, lr, load, device):
    lat_min = int(domain.split(',')[0])
    lat_max = int(domain.split(',')[1])
    lon_len = int(domain.split(',')[2])
    savename = 'DGPRN_{0}_{1}_lat{2}to{3}_lon{4}'.format(group, num, lat_min, lat_max, lon_len)
    lat_len = lat_max - lat_min
    dataloader_train = DataLoader(dataset = M1_Data.Data('train', lat_min, lat_max, lat_len, lon_len, device), batch_size = batch_size, shuffle = True, drop_last = True)
    dataloader_val = DataLoader(dataset = M1_Data.Data('val', lat_min, lat_max, lat_len, lon_len, device), batch_size = batch_size, shuffle = False, drop_last = True)
    net = M2_DGPRN.DGPRN()
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    device = torch.device(device)
    criterion = torch.nn.MSELoss(reduction = 'none')
    optimizer = torch.optim.Adam(params = net.parameters(), lr = lr)
    if load == 'True':
        netpath = abspath + '/Save/Net/DGPRN_{0}_{1}_lat{2}to{3}_lon{4}.pth'.format(group, num - 1, lat_min, lat_max, lon_len)
        net.load_state_dict(torch.load(netpath)['net'])
        optimizer.load_state_dict(torch.load(netpath)['optimizer'])
    net.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    train_loss = []
    val_loss = []
    val_output = []
    val_y = []
    val_idx = []
    meanstd = open(abspath + '/Note/MeanStd1/MeanStd_lat{0}to{1}_lon{2}.pkl'.format(lat_min, lat_max, lon_len), 'rb')
    meanstd_dic = pickle.load(meanstd)
    meanstd.close()
    sswsst_mean = np.array(meanstd_dic['sswsst_mean'])
    sswsst_std = np.array(meanstd_dic['sswsst_std'])
    endtime = time.time()
    for epoch in range(epoch_num):
        net.train()
        for batch, (x, edge_idx, y, mask_net, mask_loss, _) in enumerate(dataloader_train):
            optimizer.zero_grad()
            output, mask_loss = net(x[:, :, :11], x[:, :, 11], edge_idx, mask_net, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std, device)
            loss = criterion(output, y)

            non_zero_loss = torch.masked_select(loss, mask_loss)
            non_zero_loss = non_zero_loss.sum() / mask_loss.sum()
            non_zero_loss.backward()
            optimizer.step()
            output = output.detach()
            train_loss.append(non_zero_loss.item())
            rmse = ((torch.masked_select(loss[:, :, 0], mask_loss[:, :, 0]).sum() / mask_loss[:, :, 0].sum()).item() ** 0.5) * sswsst_std[0]
            duration = time.time() - endtime
            endtime = time.time()
            print('epoch= {0}, batch= {1}, loss= {2:.5f}, rmse= {3:.5f}, time= {4:.2f}s'.format(epoch + 1, batch + 1, non_zero_loss.item(), rmse, duration))
        if epoch_num < 100:
            net.eval()
            loss_list = []
            with torch.no_grad():
                for batch, (x, edge_idx, y, mask_net, mask_loss, idx) in enumerate(dataloader_val):
                    output, mask_loss = net(x[:, :, :11], x[:, :, 11], edge_idx, mask_net, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std, device)
                    loss = criterion(output, y)

                    non_zero_loss = torch.masked_select(loss, mask_loss)
                    non_zero_loss = non_zero_loss.sum() / mask_loss.sum()
                    loss_list.append(non_zero_loss.item())
                    rmse = ((torch.masked_select(loss[:, :, 0], mask_loss[:, :, 0]).sum() / mask_loss[:, :, 0].sum()).item() ** 0.5) * sswsst_std[0]
                    print('epoch= {0}, val_batch= {1}, val_loss= {2:.5f}, rmse= {3:.5f}'.format(epoch + 1, batch + 1, non_zero_loss.item(), rmse))
                    if epoch == epoch_num - 1:
                        output[torch.where(mask_loss == 0)] = torch.nan
                        output = output.detach().cpu().numpy()
                        val_output.append(output)
                        y[torch.where(mask_loss == 0)] = torch.nan
                        y = y.detach().cpu().numpy()
                        val_y.append(y)
                        val_idx.append(idx)
                val_loss.append(sum(loss_list) / len(loss_list))
            torch.save({'net':net.state_dict(), 'optimizer':optimizer.state_dict()}, abspath + '/Save/Net/{0}.pth'.format(savename))
        elif (epoch_num >= 100) and (epoch == epoch_num - 1):
            net.eval()
            loss_list = []
            with torch.no_grad():
                for batch, (x, edge_idx, y, mask_net, mask_loss, idx) in enumerate(dataloader_val):
                    output, mask_loss = net(x[:, :, :11], x[:, :, 11], edge_idx, mask_net, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std, device)
                    loss = criterion(output, y)

                    non_zero_loss = torch.masked_select(loss, mask_loss)
                    non_zero_loss = non_zero_loss.sum() / mask_loss.sum()
                    output = output.detach()
                    loss_list.append(non_zero_loss.item())
                    rmse = ((torch.masked_select(loss[:, :, 0], mask_loss[:, :, 0]).sum() / mask_loss[:, :, 0].sum()).item() ** 0.5) * sswsst_std[0]
                    print('epoch= {0}, val_batch= {1}, val_loss= {2:.5f}, rmse= {3:.5f}'.format(epoch + 1, batch + 1, non_zero_loss.item(), rmse))
                    output[torch.where(mask_loss == 0)] = torch.nan
                    output = output.detach().cpu().numpy()
                    val_output.append(output)
                    y[torch.where(mask_loss == 0)] = torch.nan
                    y = y.detach().cpu().numpy()
                    val_y.append(y)
                    val_idx.append(idx)
            torch.save({'net':net.state_dict(), 'optimizer':optimizer.state_dict()}, abspath + '/Save/Net/{0}.pth'.format(savename))
    val_output = np.concatenate(val_output)
    val_y = np.concatenate(val_y)
    val_idx = np.concatenate(val_idx)
    val_idx = np.concatenate([val_idx[::2], val_idx[1::2]]).reshape(2, -1)
    for var_num in range(4):
        val_output[:, :, var_num] = val_output[:, :, var_num] * sswsst_std[var_num] + sswsst_mean[var_num]
        val_y[:, :, var_num] = val_y[:, :, var_num] * sswsst_std[var_num] + sswsst_mean[var_num]
    sswd_output = 180 + np.arctan2(val_output[:, :, 1], val_output[:, :, 2]) * (180 / np.pi)
    sswd_y = 180 + np.arctan2(val_y[:, :, 1], val_y[:, :, 2]) * (180 / np.pi)
    uwnd_output = -np.multiply(val_output[:, :, 0], np.sin(sswd_output * np.pi / 180))
    vwnd_output = -np.multiply(val_output[:, :, 0], np.cos(sswd_output * np.pi / 180))
    uwnd_y = -np.multiply(val_y[:, :, 0], np.sin(sswd_y * np.pi / 180))
    vwnd_y = -np.multiply(val_y[:, :, 0], np.cos(sswd_y * np.pi / 180))
    val_output = val_output[:, :, 1:]
    val_y = val_y[:, :, 1:]
    val_output[:, :, 0] = uwnd_output
    val_output[:, :, 1] = vwnd_output
    val_y[:, :, 0] = uwnd_y
    val_y[:, :, 1] = vwnd_y
    dic = {'train_loss':train_loss, 'val_loss':val_loss, 'val_output':val_output, 'val_y':val_y, 'val_idx':val_idx}
    file = open(abspath + '/Save/Pickle/{0}.pkl'.format(savename), 'wb')
    pickle.dump(dic, file)
    file.close()

def concat(M1_Data, M2_DGPRN, device):
    loc_bt = abspath + '/DataSet/X/FY3'
    asc_list = []
    dsc_list = []
    for orbit in ['asc', 'dsc']:
    # for orbit in ['asc']:
        bt_list = loc_bt + '/{0}'.format(orbit.capitalize())
        bt_list = sorted([bt_list + '/' + i for i in os.listdir(bt_list)])
        for file_bt in bt_list:
            if file_bt.split('_')[1] == orbit:
                if orbit == 'asc':
                    asc_list.append([orbit, '{0}'.format(file_bt.split('_')[-1])])
                elif orbit == 'dsc':
                    dsc_list.append([orbit, '{0}'.format(file_bt.split('_')[-1])])
    trainval_list = asc_list + dsc_list
    test_list = []
    for data in trainval_list:
        # if data[1].split('.')[0] in ['2022', '2023']:
        if (data[1].split('.')[0] in ['2023']) and (data[1].split('.')[1] in ['07']) and (data[1].split('.')[2] in ['31']):
            test_list.append(data)
        if (data[1].split('.')[0] in ['2023']) and (data[1].split('.')[1] in ['10']) and (data[1].split('.')[2] in ['02']):
            test_list.append(data)
    test_date = []
    for test in test_list:
        test_date.append(test[1].replace('.nc', ''))
    test_date_group = []
    for _, group in groupby(test_date, key = lambda x: datetime.datetime.strptime(x, '%Y.%m.%d.%H.%M').date()):
        test_date_group.append(list(group))
    test_list_group = []
    for group in test_date_group:
        test_list_group.append([test_list[test_date.index(group_date)] for group_date in group])
    domain_list = ['0,16,24', '15,31,24', '30,46,32', '45,61,40', '60,84,24', '-16,0,24', '-31,-15,24', '-46,-30,32', '-61,-45,40', '-84,-60,24']
    print('test length= {0}'.format(len(test_list_group)))
    for concat_list in test_list_group:
        val_output_asc_all = []
        val_y_asc_all = []
        val_output_dsc_all = []
        val_y_dsc_all = []
        for domain in domain_list:
            lat_min = int(domain.split(',')[0])
            lat_max = int(domain.split(',')[1])
            lon_len = int(domain.split(',')[2])
            lat_len = lat_max - lat_min
            dataloader_val = DataLoader(dataset = M1_Data.Concat(concat_list, lat_min, lat_max, lat_len, lon_len, device), batch_size = 1, shuffle = False)
            net = M2_DGPRN.DGPRN()
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
            device = torch.device(device)
            criterion = torch.nn.MSELoss(reduction = 'none')
            netpath = abspath + '/Note/Concat/Net/DGPRN_lat{0}to{1}_lon{2}.pth'.format(lat_min, lat_max, lon_len)
            try:
                net.load_state_dict(torch.load(netpath)['net'])
            except:
                state_dict = OrderedDict()
                for k, v in torch.load(netpath)['net'].items():
                    state_dict[k[7:]] = v
                net.load_state_dict(state_dict)

            total_num = sum(p.numel() for p in net.parameters())
            print(domain, total_num)



        #     net.to(device)
        #     val_output_asc = []
        #     val_output_dsc = []
        #     val_y_asc = []
        #     val_y_dsc = []
        #     val_idx_asc = []
        #     val_idx_dsc = []
        #     meanstd = open(abspath + '/Note/MeanStd1/MeanStd_lat{0}to{1}_lon{2}.pkl'.format(lat_min, lat_max, lon_len), 'rb')
        #     meanstd_dic = pickle.load(meanstd)
        #     meanstd.close()
        #     sswsst_mean = np.array(meanstd_dic['sswsst_mean'])
        #     sswsst_std = np.array(meanstd_dic['sswsst_std'])
        #     net.eval()
        #     loss_list = []
        #     with torch.no_grad():
        #         for batch, data in enumerate(dataloader_val):
        #             output_domain = np.full([4, lat_len * 4, 360 * 4], np.nan)
        #             y_domain = np.full([4, lat_len * 4, 360 * 4], np.nan)
        #             if (np.abs(lat_min) > 75) or (np.abs(lat_max) > 75):
        #                 if len(data) == 7:
        #                     x, edge_idx, y, mask_net, mask_loss, idx, lon_start = data
        #                     x = x[0]
        #                     edge_idx = edge_idx[0]
        #                     y = y[0]
        #                     mask_net = mask_net[0]
        #                     mask_loss = mask_loss[0]
        #                     lon_start = lon_start[0]
        #                     output, mask_loss = net(x[:, :, :11], x[:, :, 11], edge_idx, mask_net, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std, device)
        #                     loss = criterion(output, y)
        #                     non_zero_loss = torch.masked_select(loss, mask_loss)
        #                     non_zero_loss = non_zero_loss.sum() / mask_loss.sum()
        #                     loss_list.append(non_zero_loss.item())
        #                     rmse = ((torch.masked_select(loss[:, :, 0], mask_loss[:, :, 0]).sum() / mask_loss[:, :, 0].sum()).item() ** 0.5) * sswsst_std[0]
        #                     print('lat{0}to{1}_lon{2}, date= {3}, rmse= {4:.5f}'.format(lat_min, lat_max, lon_len, concat_list[batch][1].replace('.nc', ''), rmse))
        #                     output[torch.where(mask_loss == 0)] = torch.nan
        #                     output = output.detach().cpu().numpy()
        #                     y[torch.where(mask_loss == 0)] = torch.nan
        #                     y = y.detach().cpu().numpy()
        #                     lon_start = lon_start.cpu().numpy().astype(int)
        #                     output = output.reshape([-1, lat_len * 4, lon_len * 4, 4]).transpose(0, 3, 1, 2)
        #                     y = y.reshape([-1, lat_len * 4, lon_len * 4, 4]).transpose(0, 3, 1, 2)
        #                     if x.shape[0] == 1:
        #                         lon_start = [lon_start]
        #                     for idx_num in range(x.shape[0]):
        #                         output_domain[:, :, np.arange(lon_start[idx_num], lon_start[idx_num] + lon_len * 4)] = output[idx_num]
        #                         y_domain[:, :, np.arange(lon_start[idx_num], lon_start[idx_num] + lon_len * 4)] = y[idx_num]
        #                 elif len(data) == 2:
        #                     idx = data
        #                     print('lat{0}to{1}_lon{2}, date= {3}, skip'.format(lat_min, lat_max, lon_len, concat_list[batch][1].replace('.nc', '')))
        #             else:
        #                 if len(data) == 7:
        #                     x, edge_idx, y, mask_net, mask_loss, idx, lon_start = data
        #                     output, mask_loss = net(x[:, :, :11], x[:, :, 11], edge_idx, mask_net, mask_loss, lat_len, lon_len, sswsst_mean, sswsst_std, device)
        #                     loss = criterion(output, y)
        #                     non_zero_loss = torch.masked_select(loss, mask_loss)
        #                     non_zero_loss = non_zero_loss.sum() / mask_loss.sum()
        #                     loss_list.append(non_zero_loss.item())
        #                     rmse = ((torch.masked_select(loss[:, :, 0], mask_loss[:, :, 0]).sum() / mask_loss[:, :, 0].sum()).item() ** 0.5) * sswsst_std[0]
        #                     print('lat{0}to{1}_lon{2}, date= {3}, rmse= {4:.5f}'.format(lat_min, lat_max, lon_len, concat_list[batch][1].replace('.nc', ''), rmse))
        #                     output[torch.where(mask_loss == 0)] = torch.nan
        #                     output = output.detach().cpu().numpy()
        #                     y[torch.where(mask_loss == 0)] = torch.nan
        #                     y = y.detach().cpu().numpy()
        #                     lon_start = int(lon_start.cpu().numpy())
        #                     output = output.reshape([lat_len * 4, lon_len * 4, 4]).transpose(2, 0, 1)
        #                     y = y.reshape([lat_len * 4, lon_len * 4, 4]).transpose(2, 0, 1)
        #                     output_domain[:, :, np.arange(lon_start, lon_start + lon_len * 4)] = output
        #                     y_domain[:, :, np.arange(lon_start, lon_start + lon_len * 4)] = y
        #                 elif len(data) == 2:
        #                     idx = data
        #                     print('lat{0}to{1}_lon{2}, date= {3}, skip'.format(lat_min, lat_max, lon_len, concat_list[batch][1].replace('.nc', '')))
        #             if idx[0][0] == 'asc':
        #                 val_output_asc.append(output_domain)
        #                 val_y_asc.append(y_domain)
        #                 val_idx_asc.append(idx[1][0][:-3])
        #             elif idx[0][0] == 'dsc':
        #                 val_output_dsc.append(output_domain)
        #                 val_y_dsc.append(y_domain)
        #                 val_idx_dsc.append(idx[1][0][:-3])
        #     if val_output_asc != []:
        #         val_output_asc = np.array(val_output_asc)
        #         val_y_asc = np.array(val_y_asc)
        #         for var_num in range(4):
        #             val_output_asc[:, var_num] = val_output_asc[:, var_num] * sswsst_std[var_num] + sswsst_mean[var_num]
        #             val_y_asc[:, var_num] = val_y_asc[:, var_num] * sswsst_std[var_num] + sswsst_mean[var_num]
        #         sswd_output_asc = 180 + np.arctan2(val_output_asc[:, 1], val_output_asc[:, 2]) * (180 / np.pi)
        #         sswd_y_asc = 180 + np.arctan2(val_y_asc[:, 1], val_y_asc[:, 2]) * (180 / np.pi)
        #         uwnd_output_asc = -np.multiply(val_output_asc[:, 0], np.sin(sswd_output_asc * np.pi / 180))
        #         vwnd_output_asc = -np.multiply(val_output_asc[:, 0], np.cos(sswd_output_asc * np.pi / 180))
        #         uwnd_y_asc = -np.multiply(val_y_asc[:, 0], np.sin(sswd_y_asc * np.pi / 180))
        #         vwnd_y_asc = -np.multiply(val_y_asc[:, 0], np.cos(sswd_y_asc * np.pi / 180))
        #         val_output_asc = val_output_asc[:, 1:]
        #         val_y_asc = val_y_asc[:, 1:]
        #         val_output_asc[:, 0] = uwnd_output_asc
        #         val_output_asc[:, 1] = vwnd_output_asc
        #         val_y_asc[:, 0] = uwnd_y_asc
        #         val_y_asc[:, 1] = vwnd_y_asc
        #         val_idx_asc = pd.to_datetime(val_idx_asc, format = '%Y.%m.%d.%H.%M')
        #         val_output_asc = xr.Dataset({'sswsst':(('time', 'var', 'lat', 'lon'), val_output_asc)}, coords = {'time':val_idx_asc, 'var':['ussw', 'vssw', 'sst'], 'lat':np.arange(lat_min, lat_max, 0.25), 'lon':np.arange(0, 360, 0.25)})['sswsst']
        #         val_y_asc = xr.Dataset({'sswsst':(('time', 'var', 'lat', 'lon'), val_y_asc)}, coords = {'time':val_idx_asc, 'var':['ussw', 'vssw', 'sst'], 'lat':np.arange(lat_min, lat_max, 0.25), 'lon':np.arange(0, 360, 0.25)})['sswsst']
        #         val_output_asc_all.append(val_output_asc)
        #         val_y_asc_all.append(val_y_asc)
        #     if val_output_dsc != []:
        #         val_output_dsc = np.array(val_output_dsc)
        #         val_y_dsc = np.array(val_y_dsc)
        #         for var_num in range(4):
        #             val_output_dsc[:, var_num] = val_output_dsc[:, var_num] * sswsst_std[var_num] + sswsst_mean[var_num]
        #             val_y_dsc[:, var_num] = val_y_dsc[:, var_num] * sswsst_std[var_num] + sswsst_mean[var_num]
        #         sswd_output_dsc = 180 + np.arctan2(val_output_dsc[:, 1], val_output_dsc[:, 2]) * (180 / np.pi)
        #         sswd_y_dsc = 180 + np.arctan2(val_y_dsc[:, 1], val_y_dsc[:, 2]) * (180 / np.pi)
        #         uwnd_output_dsc = -np.multiply(val_output_dsc[:, 0], np.sin(sswd_output_dsc * np.pi / 180))
        #         vwnd_output_dsc = -np.multiply(val_output_dsc[:, 0], np.cos(sswd_output_dsc * np.pi / 180))
        #         uwnd_y_dsc = -np.multiply(val_y_dsc[:, 0], np.sin(sswd_y_dsc * np.pi / 180))
        #         vwnd_y_dsc = -np.multiply(val_y_dsc[:, 0], np.cos(sswd_y_dsc * np.pi / 180))
        #         val_output_dsc = val_output_dsc[:, 1:]
        #         val_y_dsc = val_y_dsc[:, 1:]
        #         val_output_dsc[:, 0] = uwnd_output_dsc
        #         val_output_dsc[:, 1] = vwnd_output_dsc
        #         val_y_dsc[:, 0] = uwnd_y_dsc
        #         val_y_dsc[:, 1] = vwnd_y_dsc
        #         val_idx_dsc = pd.to_datetime(val_idx_dsc, format = '%Y.%m.%d.%H.%M')
        #         val_output_dsc = xr.Dataset({'sswsst':(('time', 'var', 'lat', 'lon'), val_output_dsc)}, coords = {'time':val_idx_dsc, 'var':['ussw', 'vssw', 'sst'], 'lat':np.arange(lat_min, lat_max, 0.25), 'lon':np.arange(0, 360, 0.25)})['sswsst']
        #         val_y_dsc = xr.Dataset({'sswsst':(('time', 'var', 'lat', 'lon'), val_y_dsc)}, coords = {'time':val_idx_dsc, 'var':['ussw', 'vssw', 'sst'], 'lat':np.arange(lat_min, lat_max, 0.25), 'lon':np.arange(0, 360, 0.25)})['sswsst']
        #         val_output_dsc_all.append(val_output_dsc)
        #         val_y_dsc_all.append(val_y_dsc)
        # if val_output_asc_all != []:
        #     val_output_asc_all = xr.concat(val_output_asc_all, dim = 'lat')
        #     val_output_asc_all = val_output_asc_all.sortby('lat')
        #     _, lat_idx = np.unique(val_output_asc_all['lat'], return_index = True)
        #     output_asc = val_output_asc_all.isel(lat = lat_idx)
        #     overlap_lat_list = [15, 30, 45, 60, -16, -31, -46, -61]
        #     for overlap_lat_start in overlap_lat_list:
        #         overlap_lat_range = np.arange(overlap_lat_start, overlap_lat_start + 1, 0.25)
        #         for overlap_lat in overlap_lat_range:
        #             output_asc.loc[:, :, overlap_lat] = val_output_asc_all.loc[:, :, overlap_lat].mean(dim = 'lat')
        #         overlap = output_asc.loc[:, :, overlap_lat_start - 3:overlap_lat_start + 4]
        #         output_asc.loc[:, :, overlap_lat_start - 1:overlap_lat_start + 2] = overlap.rolling(lat = 5, center = True).mean()[:, :, 8:-8]
        #     output_asc = output_asc.resample(time = 'D').mean(dim = 'time')
        #     slice_ice_idx = np.where(np.array(output_asc)[:, 2] < 273.15)
        #     for var_num in range(3):
        #         output_asc_temp = np.array(output_asc[:, var_num])
        #         output_asc_temp[slice_ice_idx] = np.nan
        #         output_asc[:, var_num] = output_asc_temp
        #     output_asc = output_asc.astype(dtype = np.float32)
        #     print(output_asc.shape)
        #     output_asc = xr.Dataset({'sswsst':output_asc})
        #     output_asc.to_netcdf(abspath + '/Note/Output/SWS0/SWS_asc_sswsst_{0}.nc'.format(concat_list[0][1][:-9]))
        #     val_y_asc_all = xr.concat(val_y_asc_all, dim = 'lat')
        #     val_y_asc_all = val_y_asc_all.sortby('lat')
        #     _, lat_idx = np.unique(val_y_asc_all['lat'], return_index = True)
        #     y_asc = val_y_asc_all.isel(lat = lat_idx)
        #     y_asc = y_asc.resample(time = 'D').mean(dim = 'time')
        #     for var_num in range(3):
        #         y_asc_temp = np.array(y_asc[:, var_num])
        #         y_asc_temp[slice_ice_idx] = np.nan
        #         y_asc[:, var_num] = y_asc_temp
        #     y_asc = y_asc.astype(dtype = np.float32)
        #     print(y_asc.shape)
        #     y_asc = xr.Dataset({'sswsst':y_asc})
        #     y_asc.to_netcdf(abspath + '/Note/Output/ERA50/ERA5_asc_sswsst_{0}.nc'.format(concat_list[0][1][:-9]))
        # if val_output_dsc_all != []:
        #     val_output_dsc_all = xr.concat(val_output_dsc_all, dim = 'lat')
        #     val_output_dsc_all = val_output_dsc_all.sortby('lat')
        #     _, lat_idx = np.unique(val_output_dsc_all['lat'], return_index = True)
        #     output_dsc = val_output_dsc_all.isel(lat = lat_idx)
        #     overlap_lat_list = [15, 30, 45, 60, -16, -31, -46, -61]
        #     for overlap_lat_start in overlap_lat_list:
        #         overlap_lat_range = np.arange(overlap_lat_start, overlap_lat_start + 1, 0.25)
        #         for overlap_lat in overlap_lat_range:
        #             output_dsc.loc[:, :, overlap_lat] = val_output_dsc_all.loc[:, :, overlap_lat].mean(dim = 'lat')
        #         overlap = output_dsc.loc[:, :, overlap_lat_start - 3:overlap_lat_start + 4]
        #         output_dsc.loc[:, :, overlap_lat_start - 1:overlap_lat_start + 2] = overlap.rolling(lat = 5, center = True).mean()[:, :, 8:-8]
        #     output_dsc = output_dsc.resample(time = 'D').mean(dim = 'time')
        #     slice_ice_idx = np.where(np.array(output_dsc)[:, 2] < 273.15)
        #     for var_num in range(3):
        #         output_dsc_temp = np.array(output_dsc[:, var_num])
        #         output_dsc_temp[slice_ice_idx] = np.nan
        #         output_dsc[:, var_num] = output_dsc_temp
        #     output_dsc = output_dsc.astype(dtype = np.float32)
        #     print(output_dsc.shape)
        #     output_dsc = xr.Dataset({'sswsst':output_dsc})
        #     output_dsc.to_netcdf(abspath + '/Note/Output/SWS0/SWS_dsc_sswsst_{0}.nc'.format(concat_list[0][1][:-9]))
        #     val_y_dsc_all = xr.concat(val_y_dsc_all, dim = 'lat')
        #     val_y_dsc_all = val_y_dsc_all.sortby('lat')
        #     _, lat_idx = np.unique(val_y_dsc_all['lat'], return_index = True)
        #     y_dsc = val_y_dsc_all.isel(lat = lat_idx)
        #     y_dsc = y_dsc.resample(time = 'D').mean(dim = 'time')
        #     for var_num in range(3):
        #         y_dsc_temp = np.array(y_dsc[:, var_num])
        #         y_dsc_temp[slice_ice_idx] = np.nan
        #         y_dsc[:, var_num] = y_dsc_temp
            # y_dsc = y_dsc.astype(dtype = np.float32)
            # print(y_dsc.shape)
            # y_dsc = xr.Dataset({'sswsst':y_dsc})
            # y_dsc.to_netcdf(abspath + '/Note/Output/ERA50/ERA5_dsc_sswsst_{0}.nc'.format(concat_list[0][1][:-9]))