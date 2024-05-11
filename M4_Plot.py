import os
import cmaps
import pickle
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import scipy.stats as st
import cartopy.crs as ccrs
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.gridspec as grid_spec
from scipy import stats
from matplotlib import rcParams
from global_land_mask import globe
from matplotlib.colors import LogNorm
from cartopy.util import add_cyclic_point
from matplotlib.pyplot import MultipleLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
config = {'font.family':'Times New Roman', 'font.size':12}
rcParams.update(config)
order = []
for i in range(97, 123):
    order.append(chr(i))

def rgb_to_hex(rgb):
    RGB = rgb.split(',')
    color = '#'
    for i in RGB:
        num = int(i)
        color = color + str(hex(num))[-2:].replace('x', '0').upper()
    return color

def plot(group, num, domain, idx):
    def plot_loss(dic, rmse_u, rmse_v, rmse_t, rmse_w, rmse_dir):
        plt.figure(figsize = (10, 5))
        ax1 = plt.gca()
        ax1.plot(np.arange(1, len(dic['train_loss']) + 1), dic['train_loss'], marker = 'o', color = rgb_to_hex('075,101,176'), label = 'Train Loss')
        ax2 = ax1.twiny()
        ax2.plot(np.arange(1, len(dic['val_loss']) + 1), dic['val_loss'], marker = 'o', color = rgb_to_hex('244,111,068'))
        ax1.plot(np.nan, np.nan, marker = 'o', color = rgb_to_hex('244,111,068'), \
                 label = 'Validation Loss\nRMSE U = {0:.5f} m/s\nRMSE V = {1:.5f} m/s\nRMSE T = {2:.5f} K\nRMSE W = {3:.5f} m/s\nRMSE Dir = {4:.5f} °'.format(rmse_u, rmse_v, rmse_t, rmse_w, rmse_dir))
        ax1.legend(bbox_to_anchor = (1.05, 0), loc = 3, borderaxespad = 0, frameon = False)
        ax1.set_xlabel('Batch Num', fontsize = 15)
        ax2.set_xlabel('Epoch', fontsize = 15)
        ax1.set_ylabel('Loss', fontsize = 15)
        ax1.set_xlim(1, len(dic['train_loss']))
        ax2.set_xlim(0, len(dic['val_loss']))
        # plt.ylim(0.04, 0.1)
        plt.savefig(abspath + '/Save/Plot/{0}/Loss.png'.format(savename), dpi = 1000, bbox_inches = 'tight')
    def plot_bt(title, lat, lon, var):
        if (np.min(lon) < 0) and (np.max(lon) > 0):
            lon = lon - 90
            central_longitude = 90
        else:
            central_longitude = 0
        fig = plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = central_longitude))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
        ax.add_feature(cfeature.LAND.with_scale('10m'), color = 'lightgrey')
        ax.set_xticks(np.arange(int(np.min(lon)), int(np.min(lon)) + lon_len + 0.1, 4))
        ax.set_yticks(np.arange(int(np.min(lat)), np.max(lat), 3))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
        pcolor = plt.scatter(lon, lat, c = var, transform = ccrs.PlateCarree(central_longitude = central_longitude), cmap = cmaps.BlueDarkRed18, s = 18, marker = 's')
        fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'K')
        plt.title(title, loc = 'center', fontsize = 25)
        plt.savefig(abspath + '/Save/Plot/{0}/{1}.png'.format(savename, title), dpi = 1000, bbox_inches = 'tight')
    def plot_uvt(title, lat, lon, var, colorbar_label, pcolor_vmin, pcolor_vmax):
        fig = plt.figure(figsize = (8, 8))
        if np.max(lon) > 180:
            lon = lon - 90
            central_longitude = 90
        else:
            central_longitude = 0
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = central_longitude))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
        ax.add_feature(cfeature.LAND.with_scale('10m'), color = 'lightgrey')
        ax.set_xticks(np.arange(int(lon[0]), lon[-1] + 0.1, 4))
        ax.set_yticks(np.arange(int(lat[0]), lat[-1], 3))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
        pcolor = plt.pcolor(lon, lat, var, transform = ccrs.PlateCarree(central_longitude = central_longitude), cmap = cmaps.BlueDarkRed18, vmin = pcolor_vmin, vmax = pcolor_vmax)
        fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = colorbar_label)
        plt.title(title, loc = 'center', fontsize = 25)
        plt.savefig(abspath + '/Save/Plot/{0}/{1}.png'.format(savename, title), dpi = 1000, bbox_inches = 'tight')
    def plot_wind(title, lat, lon, uwnd, vwnd, wind, pcolor_vmin, pcolor_vmax):
        quiver_interval = lon_len // 12
        fig = plt.figure(figsize = (8, 8))
        if np.max(lon) > 180:
            lon = lon - 90
            central_longitude = 90
        else:
            central_longitude = 0
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = central_longitude))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
        ax.add_feature(cfeature.LAND.with_scale('10m'), color = 'lightgrey')
        ax.set_xticks(np.arange(int(lon[0]), lon[-1] + 0.1, 4))
        ax.set_yticks(np.arange(int(lat[0]), lat[-1], 3))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
        pcolor = plt.pcolor(lon, lat, wind, transform = ccrs.PlateCarree(central_longitude = central_longitude), cmap = cmaps.BlueDarkRed18, vmin = pcolor_vmin, vmax = pcolor_vmax)
        plt.quiver(lon[::quiver_interval], lat[::quiver_interval], uwnd[::quiver_interval, ::quiver_interval], vwnd[::quiver_interval, ::quiver_interval], transform = ccrs.PlateCarree(central_longitude = central_longitude))
        fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'm/s')
        plt.title(title, loc = 'center', fontsize = 25)
        plt.savefig(abspath + '/Save/Plot/{0}/{1}.png'.format(savename, title), dpi = 1000, bbox_inches = 'tight')
    lat_min = int(domain.split(',')[0])
    lat_max = int(domain.split(',')[1])
    lon_len = int(domain.split(',')[2])
    lat_len = lat_max - lat_min
    savename = 'DGPRN_{0}_{1}_lat{2}to{3}_lon{4}'.format(group, num, lat_min, lat_max, lon_len)
    plotdir = abspath + '/Save/Plot/{0}'.format(savename)
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    loc_bt = abspath + '/DataSet/X/FY3'
    loc_sswsst = abspath + '/DataSet/Y/ERA5'
    file = open(abspath + '/Save/Pickle/{0}.pkl'.format(savename), 'rb')
    dic = pickle.load(file)
    file.close()
    bt = xr.open_dataset(loc_bt + '/{0}/FY3_{1}_bt_{2}'.format(dic['val_idx'][0, idx].split('_')[0].capitalize(),dic['val_idx'][0, idx].split('_')[0], dic['val_idx'][1, idx]))
    era5 = xr.open_dataset(loc_sswsst + '/{0}/ERA5_{1}_sswsst_{2}'.format(dic['val_idx'][0, idx].split('_')[0].capitalize(),dic['val_idx'][0, idx].split('_')[0], dic['val_idx'][1, idx]))
    era5_lat = np.array(era5['lat'])
    era5_lon = np.array(era5['lon'])
    bt = np.array(bt['bt'])
    era5 = np.array(era5['sswsst'])
    slice_idx = np.where((era5_lat >= lat_min) & (era5_lat < lat_max))[0]
    bt = bt[:, slice_idx]
    era5_lat = era5_lat[slice_idx]
    era5_lon = era5_lon[slice_idx]
    slice_idx = np.where((bt <= 10) | (bt >= 310))[1]
    slice_idx = np.setdiff1d(np.arange(bt.shape[1]), slice_idx)
    bt = bt[:, slice_idx]
    era5_lat = era5_lat[slice_idx]
    era5_lon = era5_lon[slice_idx]
    if (np.abs(lat_min) > 75) or (np.abs(lat_max) > 75):
        ramdom_lon = float(dic['val_idx'][0, idx].split('_')[1])
        ramdom_lon_start = ramdom_lon - 12
        ramdom_lon_end = ramdom_lon + 12
        if ramdom_lon_start < -180:
            ramdom_lon_start = ramdom_lon_start + 360
            slice_idx = np.where((era5_lon <= ramdom_lon_end) | (era5_lon > ramdom_lon_start))[0]
        elif ramdom_lon_end > 180:
            ramdom_lon_end = ramdom_lon_end - 360
            slice_idx = np.where((era5_lon >= ramdom_lon_start) | (era5_lon < ramdom_lon_end))[0]
        else:
            slice_idx = np.where((era5_lon >= ramdom_lon_start) & (era5_lon < ramdom_lon_end))[0]
        bt = bt[:, slice_idx]
        era5_lat = era5_lat[slice_idx]
        era5_lon = era5_lon[slice_idx]
    era5_lon[np.where(era5_lon < 0)] = 360 + era5_lon[np.where(era5_lon < 0)]
    lat = np.arange(lat_min, lat_max, 0.25)
    lon = np.arange(np.min(era5_lon), np.min(era5_lon) + lon_len, 0.25)
    sws = dic['val_output']
    era5 = dic['val_y']
    sws_wind = (sws[:, :, 0] ** 2 + sws[:, :, 1] ** 2) ** 0.5
    era5_wind = (era5[:, :, 0] ** 2 + era5[:, :, 1] ** 2) ** 0.5
    sws_dir = 180 + np.arctan2(sws[:, :, 0], sws[:, :, 1]) * (180 / np.pi)
    era5_dir = 180 + np.arctan2(era5[:, :, 0], era5[:, :, 1]) * (180 / np.pi)
    sws_dir = sws_dir[np.where(era5_wind >= 10)]
    era5_dir = era5_dir[np.where(era5_wind >= 10)]
    error_dir = np.abs(sws_dir - era5_dir)
    adjust_error_dir_idx = np.where(error_dir >= 180)
    error_dir[adjust_error_dir_idx] = 360 - error_dir[adjust_error_dir_idx]
    print('Mean Val Loss= {0:.5f}'.format(np.mean(dic['val_loss'])))
    rmse_u = np.nanmean((sws[:, :, 0] - era5[:, :, 0]) ** 2) ** 0.5
    rmse_v = np.nanmean((sws[:, :, 1] - era5[:, :, 1]) ** 2) ** 0.5
    rmse_t = np.nanmean((sws[:, :, 2] - era5[:, :, 2]) ** 2) ** 0.5
    rmse_w = np.nanmean((sws_wind - era5_wind) ** 2) ** 0.5
    rmse_dir = np.nanmean(error_dir ** 2) ** 0.5
    print('RMSE_U= {0:.5f} m/s'.format(rmse_u))
    print('RMSE_V= {0:.5f} m/s'.format(rmse_v))
    print('RMSE_T= {0:.5f} K'.format(rmse_t))
    print('RMSE_W= {0:.5f} m/s'.format(rmse_w))
    print('RMSE_Dir= {0:.5f} °'.format(rmse_dir))
    sws = sws[idx].reshape([lat_len * 4, lon_len * 4, 3])
    era5 = era5[idx].reshape([lat_len * 4, lon_len * 4, 3])
    sws_wind = sws_wind[idx].reshape([lat_len * 4, lon_len * 4])
    era5_wind = era5_wind[idx].reshape([lat_len * 4, lon_len * 4])
    plot_loss(dic, rmse_u, rmse_v, rmse_t, rmse_w, rmse_dir)
    plot_bt('BT', era5_lat, era5_lon, bt[4])
    plot_uvt('DGPRN_U', lat, lon, sws[:, :, 0], 'm/s', np.nanmin(era5[:, :, 0]), np.nanmax(era5[:, :, 0]))
    plot_uvt('DGPRN_V', lat, lon, sws[:, :, 1], 'm/s', np.nanmin(era5[:, :, 1]), np.nanmax(era5[:, :, 1]))
    plot_uvt('DGPRN_T', lat, lon, sws[:, :, 2], 'K', np.nanmin(era5[:, :, 2]), np.nanmax(era5[:, :, 2]))
    plot_uvt('ERA5_U', lat, lon, era5[:, :, 0], 'm/s', np.nanmin(era5[:, :, 0]), np.nanmax(era5[:, :, 0]))
    plot_uvt('ERA5_V', lat, lon, era5[:, :, 1], 'm/s', np.nanmin(era5[:, :, 1]), np.nanmax(era5[:, :, 1]))
    plot_uvt('ERA5_T', lat, lon, era5[:, :, 2], 'K', np.nanmin(era5[:, :, 2]), np.nanmax(era5[:, :, 2]))
    plot_wind('DGPRN_W', lat, lon, sws[:, :, 0], sws[:, :, 1], sws_wind, np.nanmin(era5_wind), np.nanmax(era5_wind))
    plot_wind('ERA5_W', lat, lon, era5[:, :, 0], era5[:, :, 1], era5_wind, np.nanmin(era5_wind), np.nanmax(era5_wind))
def domain():
    loc = abspath + '/Save/Plot/Domain'
    plotdir = loc + '/PDF'
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    for orbit in ['asc', 'dsc']:
        file_list = abspath + '/DataSet/X/FY3/{0}'.format(orbit.capitalize())
        file_list = sorted([file_list + '/' + i for i in os.listdir(file_list)])[::2000]
    # file_list = file_list[:2]
    ave_list = []
    std_list = []
    pdf_x_list = []
    pdf_list = []
    x_list = []
    y_list = []
    # area_list = [[-45, -30], [-30, -15], [-15, 0], [0, 15], [15, 30], [30, 45]]
    # area_list = [[-75, 75], [-60, -45], [-45, -30], [-30, -15], [-15, 0], [0, 15], [15, 30], [30, 45], [45, 60], [60, 75]]
    # area_list = [[-84, 84], [-84, -60], [-60, -45], [-45, -30], [-30, -15], [-15, 0], [0, 15], [15, 30], [30, 45], [45, 60], [60, 84]]
    area_list = [[-84, 84], [60, 84], [45, 60], [30, 45], [15, 30], [0, 15], [-15, 0], [-30, -15], [-45, -30], [-60, -45], [-84, -60]]
    for area in area_list:
        print(area)
        var_list = []
        for file in file_list:
            file = xr.open_dataset(file)
            lat = file['lat']
            bt = file['bt']
            slice_idx = np.where((lat >= area[0]) & (lat < area[1]))[0]
            # var_list.append(np.array(xr.open_dataset(file)['bt'][5]).flatten())
            var_list.append(np.array(bt[0, slice_idx]).flatten())

        var = np.concatenate(var_list).flatten()
        var = sorted(var)
        # ave = np.mean(var)
        # std = np.std(var)
        pdf = np.histogram(var, bins = 'auto', density = True)
        interval = (pdf[1][1] - pdf[1][0]) / 2
        pdf_x = []
        for i in range(len(pdf[1])):
            pdf_x.append(pdf[1][i])
        pdf_x.append(pdf[1][-1] + interval)
        pdf = list(pdf[0])
        pdf.insert(0, 0)
        pdf.append(0)
        # x = np.arange(ave - 5 * std, ave + 5 * std, 0.01)
        # min_error = 1
        # true_peak = 1
        # for peak in np.linspace(np.max(pdf) - 0.1, np.max(pdf) + 0.1, 1000):
        #     error = np.max(pdf) - np.max(st.norm(loc = ave, scale = 1 / (peak * ((2 * np.pi) ** 1/2))).pdf(x))
        #     if np.abs(error) < np.abs(min_error):
        #         min_error = error
                # true_peak = peak
        # y = st.norm(loc = ave, scale = 1 / (true_peak * ((2 * np.pi) ** 1/2))).pdf(x)
        # ave_list.append(ave)
        # std_list.append(std)
        pdf_x_list.append(pdf_x)
        pdf_list.append(np.array(pdf) * 100)
        # x_list.append(x)
        # y_list.append(np.array(y) * 100)


    # colors = ['219,049,036', '252,140,090', '255,223,146', '230,241,243', '144,190,224', '075,116,178']
    colors = ['128,116,200', '120,149,193', '168,203,223', '214,239,244', '242,250,252', '247,251,201', '245,235,174', '240,194,132', '239,139,103', '227,098,093', '181,071,100', '153,034,036']
    gs = grid_spec.GridSpec(len(pdf_x_list), 1)
    fig = plt.figure(figsize = (10, 9))
    for pdf_x_num in range(len(pdf_x_list)):
        ax = fig.add_subplot(gs[pdf_x_num:pdf_x_num + 1, 0:])
        ax.plot(pdf_x_list[pdf_x_num], pdf_list[pdf_x_num], color = 'grey', lw = 0.5)
        ax.fill_between(pdf_x_list[pdf_x_num], pdf_list[pdf_x_num], color = rgb_to_hex(colors[pdf_x_num]), alpha = 1)
        ax.axhline(y = 0, xmin = 0, xmax = 1, color = 'black', lw = 1)
        ax.set_xlim(150, 175)
        ax.set_ylim(0, 22)

        # make background transparent
        rect = ax.patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        # ax.set_yticklabels([])
        
        ax.set_yticks([])

        if pdf_x_num == len(pdf_x_list) - 1:
            ax.set_xlabel('Brightness Temperature (K)', fontsize = 30)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        for s in ['top', 'right', 'left', 'bottom']:
            ax.spines[s].set_visible(False)
        # ax.text(149.5, 0, '{0}° to {1}°'.format(area_list[pdf_x_num][0], area_list[pdf_x_num][1]), fontsize = 10, ha = 'right')
        if pdf_x_num == 0:
            ax.text(148, 0, 'Global', color = 'black', fontsize = 30, fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'center')
        else:
            ax.text(148, 0, pdf_x_num, color = 'black', fontsize = 30, fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'center')
        if pdf_x_num == 0:
            ax.set_title('Probability density distribution', loc = 'center', fontsize = 30, pad = -50)
    gs.update(hspace = -0.6)
    plt.gca().tick_params(labelsize = 25)
    # plt.text((150 + 175) / 2, 0.85, 'Probability density distribution', fontsize = 30)

    # plt.tight_layout()
    plt.savefig(plotdir + '/PDF.png', dpi = 1000, bbox_inches = 'tight')



    # plt.figure(figsize = (14, 10))
    # plt.plot(pdf_x_list[0], pdf_list[0], color = 'steelblue', label = 'FY-4A Sample Probability Distribution', linestyle = '-', linewidth = 7)
    # # plt.plot(pdf_x_list[1], pdf_list[1], color = 'darkred', label = 'HMW8 Sample Probability Distribution', linestyle = '--', linewidth = 7)
    # plt.plot(x_list[0], y_list[0], color = 'grey', alpha = 0.3, label = 'Normal Distribution', linewidth = 7, zorder = 10)
    # # plt.plot(x_list[1], y_list[1], color = 'grey', alpha = 0.3, linewidth = 7, zorder = 20)
    # plt.axvline(ave_list[0], ymax = 1, color = 'grey', linestyle = '--', alpha = 0.5)
    # # plt.axvline(ave_list[1], ymax = 1, color = 'grey', linestyle = '--', alpha = 0.5)
    # plt.legend(loc = [1 + (2 / 35), 0.7], frameon = False, fontsize = 32)
    # plt.rc('legend', fontsize = 16)
    # # plt.text(x = 168, y = 0.7, s = 'FY-4A Sample Mean: {0}\nHMW8 Sample Mean: {1}\nFY-4A Sample Standard Deviation: {2}\nHMW8 Sample Standard Deviation: {3}' \
    # #          .format(round(ave_list[0], 2), round(ave_list[1], 2), round(std_list[0], 2), round(std_list[1], 2)), fontsize = 32)
    # # plt.xlim(130, 165)
    # # plt.ylim(0, 10)
    # # plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    # # plt.gca().yaxis.set_major_locator(MultipleLocator(2))
    # plt.xlabel('Pixel', fontsize = 32)
    # plt.ylabel('Probability Density /%', fontsize = 32)
    # plt.tick_params(pad = 7, labelsize = 30)
    # plt.savefig(plotdir + '/PDF_Overlapped.png', dpi = 1000, bbox_inches = 'tight')

def flow():
    domain_list = [[60, 84, 24], [45, 60, 40], [30, 45, 32], [15, 30, 24], [0, 15, 24], [-15, 0, 24], [-30, -15, 24], [-45, -30, 32], [-60, -45, 40], [-84, -60, 24]]

    def plot_input_bt(bt_data):
        bt = np.array(bt_data['bt'])
        lat = np.array(bt_data['lat'])
        for channel_num in range(10):
            lat_idx = ((lat + 90) * 8).astype(int)
            lon_idx = np.array(bt_data['lon'])
            lon_idx[np.where(lon_idx < 0)] = 360 + lon_idx[np.where(lon_idx < 0)]
            lon_idx = (lon_idx * 8).astype(int)
            bt_domain = np.full([180 * 8 + 1, 360 * 8 + 1], np.nan)
            bt_domain[lat_idx, lon_idx] = bt[channel_num]
            plt.figure(figsize = (8, 8))
            ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
            ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
            ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree(central_longitude = 180))
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            plt.pcolor(np.arange(0, 360.01, 0.125), np.arange(-90, 90.01, 0.125), bt_domain, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18)
            plt.savefig(plotdir + '/Input_Channel{0}.png'.format(channel_num + 1), dpi = 1000, bbox_inches = 'tight')

    def plot_input_saa(bt_data):
        saa = np.array(bt_data['saa'])
        lat = np.array(bt_data['lat'])
        lat_idx = ((lat + 90) * 8).astype(int)
        lon_idx = np.array(bt_data['lon'])
        lon_idx[np.where(lon_idx < 0)] = 360 + lon_idx[np.where(lon_idx < 0)]
        lon_idx = (lon_idx * 8).astype(int)
        saa_domain = np.full([180 * 8 + 1, 360 * 8 + 1], np.nan)
        saa_domain[lat_idx, lon_idx] = saa[-1]
        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree(central_longitude = 180))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        plt.pcolor(np.arange(0, 360.01, 0.125), np.arange(-90, 90.01, 0.125), saa_domain, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18, vmin = 150, vmax = 151.5)
        plt.savefig(plotdir + '/Input_SAA.png', dpi = 1000, bbox_inches = 'tight')

    def plot_input_topography():
        topography = np.array(xr.open_dataset(abspath + '/DataSet/X/SHP/SHP_topography.nc')['topography'])
        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree(central_longitude = 180))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        level_list = np.array([-8000, -6000, -4000, -2000, -1000, -200, -50, 0, 50, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
        level_list = (level_list + 10757.00000001) / (7182.93772189 + 10757.00000001)
        color_list = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#006837', '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f7fcb9', '#c9bc87', '#a69165', '#856b49', '#664830', '#ad9591', '#d7ccca']
        plt.contourf(np.arange(-180, 180.01, 0.125), np.arange(90, -90.01, -0.125), topography, transform = ccrs.PlateCarree(), levels = level_list, colors = color_list)
        plt.savefig(plotdir + '/Input_Topography.png', dpi = 1000, bbox_inches = 'tight')

    def plot_positional_encoding(bt_data):
        domain = domain_list[1]
        lat_min = domain[0]
        lat_max = domain[1]
        bt = np.array(bt_data['bt'])
        lat = np.array(bt_data['lat'])
        lon = np.array(bt_data['lon'])
        slice_domain_idx = np.where((lat >= lat_min) & (lat < lat_max + 1))[0]
        bt = bt[:, slice_domain_idx]
        lat = lat[slice_domain_idx]
        lon = lon[slice_domain_idx]
        lat_idx = ((lat + 90) * 8).astype(int)
        lon[np.where(lon < 0)] = 360 + lon[np.where(lon < 0)]
        lon_idx = (lon * 8).astype(int)
        # bt_domain = np.full([180 * 8 + 1, 360 * 8 + 1], np.nan)
        # bt_domain[lat_idx, lon_idx] = bt[9]
        # plt.figure(figsize = (8, 8))
        # ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        # ax.set_extent([np.min(lon) + 5 + 0.0625, np.min(lon) + 7 - 0.0625, lat_min + 10 + 0.0625, lat_min + 12 - 0.0625])
        # ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.3)
        # ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        # ax.xaxis.set_major_formatter(LongitudeFormatter())
        # ax.yaxis.set_major_formatter(LatitudeFormatter())
        # plt.pcolor(np.arange(-180, 180.01, 0.125), np.arange(-90, 90.01, 0.125), bt_domain, transform = ccrs.PlateCarree(central_longitude = 180), cmap = cmaps.BlueDarkRed18)
        # for i in np.arange(np.min(lon) + 5 - 180, np.min(lon) + 7 - 180, 0.125):
        #     plt.axvline(i - 0.0625, color = 'black')
        # for i in np.arange(lat_min + 10, lat_min + 12, 0.125):
        #     plt.axhline(i - 0.0625, color = 'black')
        # for i in np.arange(np.min(lon) + 5 - 180, np.min(lon) + 7 - 180, 0.125):
        #     for j in np.arange(lat_min + 10, lat_min + 12, 0.125):
        #         if (i != np.min(lon) + 5 - 180) and (j != lat_min + 10):
        #             if np.isnan(bt_domain[int((j + 90) * 8), int((i + 180) * 8)]) == False:
        #                 plt.text(i, j - 0.01, '1', color = 'black', fontsize = 30, fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'center')
        #             else:
        #                 plt.text(i, j - 0.01, '0', color = 'black', fontsize = 30, fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'center')
        # plt.savefig(plotdir + '/Positional_Encoding.png', dpi = 1000, bbox_inches = 'tight')

        for channel_num in range(10):
            bt_domain = np.full([180 * 8 + 1, 360 * 8 + 1], np.nan)
            bt_domain[lat_idx, lon_idx] = bt[channel_num]
            plt.figure(figsize = (8, 8))
            ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
            ax.set_extent([np.min(lon), np.min(lon) + 32, lat_min, lat_max])
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.3)
            ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            plt.pcolor(np.arange(-180, 180.01, 0.125), np.arange(-90, 90.01, 0.125), bt_domain, transform = ccrs.PlateCarree(central_longitude = 180), cmap = cmaps.BlueDarkRed18)
            # plt.fill_between([np.min(lon) + 5 + 0.0625 - 180, np.min(lon) + 7 - 0.0625 - 180], lat_min + 10 + 0.0625, lat_min + 12 - 0.0625, color = 'none', edgecolor = rgb_to_hex('060,064,091'), linewidth = 5)
            plt.savefig(plotdir + '/BT_Domain_{0}.png'.format(channel_num), dpi = 1000, bbox_inches = 'tight')

        # plt.figure(figsize = (8, 8))
        # ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        # ax.set_extent([np.min(lon), np.min(lon) + 32, lat_min, lat_max])
        # ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.3)
        # ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        # ax.xaxis.set_major_formatter(LongitudeFormatter())
        # ax.yaxis.set_major_formatter(LatitudeFormatter())
        # for i in range(len(lon)):
        #     if globe.is_land(np.around(lat[i]), -(np.around(lon[i]) - 180)) == False:
        #         plt.scatter(np.around(lon[i]), np.around(lat[i]), color = 'grey', edgecolors = 'black', marker = 'o', s = 15, transform = ccrs.PlateCarree())
        # plt.fill_between([np.min(lon) + 5 + 0.0625 - 180, np.min(lon) + 7 - 0.0625 - 180], lat_min + 10 + 0.0625, lat_min + 12 - 0.0625, color = 'none', edgecolor = rgb_to_hex('060,064,091'), linewidth = 5)
        # plt.savefig(plotdir + '/Node.png', dpi = 1000, bbox_inches = 'tight')

    def plot_domain(bt_data):
        lon_min_list_all = []
        for domain in domain_list:
            lon_min_list = []
            lat_min = domain[0]
            lat_max = domain[1]
            lon_len = domain[2]
            bt = np.array(bt_data['bt'])
            lat = np.array(bt_data['lat'])
            lon = np.array(bt_data['lon'])
            lon[np.where(lon < 0)] = 360 + lon[np.where(lon < 0)]
            slice_domain_idx = np.where((lat >= lat_min) & (lat < lat_max))[0]
            lon = lon[slice_domain_idx]
            lon_min_list.append(np.min(lon))
            while_num = 0
            while (len(lon) > 0) or (while_num == 0):
                while_num = while_num + 1
                slice_domain_idx = np.where(lon > np.min(lon) + lon_len)[0]
                lon = lon[slice_domain_idx]
                if len(lon) > 0:
                    lon_min_list.append(np.min(lon))
            lon_min_list_all.append(lon_min_list)
        lat_idx = ((lat + 90) * 8).astype(int)
        lon_idx = np.array(bt_data['lon'])
        lon_idx[np.where(lon_idx < 0)] = 360 + lon_idx[np.where(lon_idx < 0)]
        lon_idx = (lon_idx * 8).astype(int)
        bt_domain = np.full([180 * 8 + 1, 360 * 8 + 1], np.nan)
        bt_domain[lat_idx, lon_idx] = bt[9]
        fig = plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree(central_longitude = 180))
        ax.set_xticks(range(-180, 180 + 1, 60))
        ax.set_yticks([-84, -60, -45, -30, -15, 0, 15, 30, 45, 60, 84])
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis = 'both', which = 'major', labelsize = 25, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
        for domain_num in range(len(domain_list)):
            lat_min = domain_list[domain_num][0]
            lat_max = domain_list[domain_num][1]
            lon_len = domain_list[domain_num][2]
            for lon_min in lon_min_list_all[domain_num]:
                if domain_num == 1:
                    plt.fill_between([lon_min - 180, lon_min + lon_len - 180], lat_min, lat_max, color = rgb_to_hex('244,111,068'), alpha = 0.3, zorder = 10)
                    plt.fill_between([lon_min - 180, lon_min + lon_len - 180], lat_min, lat_max, color = 'none', edgecolor = rgb_to_hex('060,064,091'), linewidth = 0.7, zorder = 20)
                else:
                    plt.fill_between([lon_min - 180, lon_min + lon_len - 180], lat_min, lat_max, color = rgb_to_hex('130,178,154'), alpha = 0.3, zorder = 10)
                    plt.fill_between([lon_min - 180, lon_min + lon_len - 180], lat_min, lat_max, color = 'none', edgecolor = rgb_to_hex('060,064,091'), linewidth = 0.7, zorder = 20)
            plt.text(170, (lat_min + lat_max) / 2 - 1.9, domain_num + 1, fontsize = 23, fontweight = 'bold', horizontalalignment = 'center', verticalalignment = 'center')
            plt.axhline(lat_min, color = 'grey', linewidth = 0.5, linestyle = '--', zorder = 10)
        plt.axhline(84, color = 'grey', linewidth = 0.5, linestyle = '--', zorder = 10)
        pcolor = plt.pcolor(np.arange(0, 360.01, 0.125), np.arange(-90, 90.01, 0.125), bt_domain, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18)
        plt.savefig(plotdir + '/Divide_Domain.png', dpi = 1000, bbox_inches = 'tight')

    def plot_edge(bt_data):
        domain = domain_list[1]
        lat_min = domain[0]
        lat_max = domain[1]
        lon_len = domain[2]
        lat_len = lat_max - lat_min
        bt = np.array(bt_data['bt'])
        lat = np.array(bt_data['lat'])
        lon = np.array(bt_data['lon'])
        slice_domain_idx = np.where((lat >= lat_min) & (lat < lat_max + 1))[0]
        bt = bt[:, slice_domain_idx]
        lat = lat[slice_domain_idx]
        lon = lon[slice_domain_idx]
        linear = open(abspath + '/Note/LinearRegression/LinearRegression_lat{0}to{1}_lon{2}.pkl'.format(lat_min, lat_max + 1, lon_len), 'rb')
        linear_dic = pickle.load(linear)
        linear.close()
        linear = [np.array(linear_dic['a']).reshape(-1, 1), np.array(linear_dic['b'])]
        sst = (np.dot(bt.transpose(1, 0), linear[0]) + linear[1])[:, 0]
        slice_ice_idx = np.where(sst >= 273.15)
        sst = sst[slice_ice_idx]
        lat = lat[slice_ice_idx]
        lon = lon[slice_ice_idx]
        lat_idx = ((lat + 90) * 8).astype(int)
        lon[np.where(lon < 0)] = 360 + lon[np.where(lon < 0)]
        lon_idx = (lon * 8).astype(int)
        sst_domain = np.full([180 * 8 + 1, 360 * 8 + 1], np.nan)
        sst_domain[lat_idx, lon_idx] = sst
        lat_grid = np.arange(-90, 90.01, 0.125)
        lon_grid = np.arange(-180, 180.01, 0.125)
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
        dx, dy = mpcalc.lat_lon_grid_deltas(lon_grid, lat_grid)
        sst_gradient_domain = np.array(mpcalc.gradient(sst_domain, deltas = (dy, dx)))
        sst_gradient_domain = ((sst_gradient_domain[0] ** 2 + sst_gradient_domain[1] ** 2) ** 0.5) * 1000
        sst_gradient_domain[0] = np.nan
        sst_gradient_domain[-1] = np.nan
        distance_domain = (-250) * sst_gradient_domain + 40
        distance_domain[np.where(distance_domain < 15)] = 15

        # plt.figure(figsize = (8, 8))
        # ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        # ax.set_extent([np.min(lon), np.min(lon) + 32, lat_min, lat_max])
        # ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.3)
        # ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        # ax.xaxis.set_major_formatter(LongitudeFormatter())
        # ax.yaxis.set_major_formatter(LatitudeFormatter())
        # plt.pcolor(np.arange(-180, 180.01, 0.125), np.arange(-90, 90.01, 0.125), sst_domain, transform = ccrs.PlateCarree(central_longitude = 180), cmap = cmaps.BlueDarkRed18)
        # plt.savefig(plotdir + '/Linear_SST.png', dpi = 1000, bbox_inches = 'tight')

        # plt.figure(figsize = (8, 8))
        # ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        # ax.set_extent([np.min(lon), np.min(lon) + 32, lat_min, lat_max])
        # ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.3)
        # ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        # ax.xaxis.set_major_formatter(LongitudeFormatter())
        # ax.yaxis.set_major_formatter(LatitudeFormatter())
        # plt.pcolor(np.arange(-180, 180.01, 0.125), np.arange(-90, 90.01, 0.125), sst_gradient_domain, transform = ccrs.PlateCarree(central_longitude = 180), vmin = 0, vmax = 0.1, cmap = cmaps.sunshine_9lev)
        # plt.savefig(plotdir + '/Gradient.png', dpi = 1000, bbox_inches = 'tight')

        # plt.figure(figsize = (8, 8))
        # ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        # ax.set_extent([np.min(lon), np.min(lon) + 32, lat_min, lat_max])
        # ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.3)
        # ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        # ax.xaxis.set_major_formatter(LongitudeFormatter())
        # ax.yaxis.set_major_formatter(LatitudeFormatter())
        # plt.pcolor(np.arange(-180, 180.01, 0.125), np.arange(-90, 90.01, 0.125), distance_domain, transform = ccrs.PlateCarree(central_longitude = 180), vmin = 15, vmax = 45, cmap = cmaps.sunshine_9lev)
        # plt.savefig(plotdir + '/Distance.png', dpi = 1000, bbox_inches = 'tight')

        value_idx = np.where(np.isnan(distance_domain) == False)
        distance = distance_domain[np.min(value_idx[0]) - 1:np.min(value_idx[0]) + (lat_len + 1) * 8 - 1, np.min(value_idx[1]) - 4:np.min(value_idx[1]) + (lon_len * 8) - 4]
        sst_grid = sst_domain[np.min(value_idx[0]) - 1:np.min(value_idx[0]) + (lat_len + 1) * 8 - 1, np.min(value_idx[1]) - 4:np.min(value_idx[1]) + (lon_len * 8) - 4]
        grid_idx = np.arange(16 * 8 * lon_len * 8).reshape(16 * 8, lon_len * 8)
        distance_1 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_1.pkl'.format(lat_min, lat_max + 1, lon_len), 'rb')
        distance_dic_1 = pickle.load(distance_1)
        distance_1.close()
        distance_1 = np.array(distance_dic_1['distance'])
        distance_2 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_2.pkl'.format(lat_min, lat_max + 1, lon_len), 'rb')
        distance_dic_2 = pickle.load(distance_2)
        distance_2.close()
        distance_2 = np.array(distance_dic_2['distance'])
        distance_3 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_3.pkl'.format(lat_min, lat_max + 1, lon_len), 'rb')
        distance_dic_3 = pickle.load(distance_3)
        distance_3.close()
        distance_3 = np.array(distance_dic_3['distance'])
        distance_4 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_4.pkl'.format(lat_min, lat_max + 1, lon_len), 'rb')
        distance_dic_4 = pickle.load(distance_4)
        distance_4.close()
        distance_4 = np.array(distance_dic_4['distance'])
        distance_5 = open(abspath + '/Note/DistanceMatrix/Distance_lat{0}to{1}_lon{2}_5.pkl'.format(lat_min, lat_max + 1, lon_len), 'rb')
        distance_dic_5 = pickle.load(distance_5)
        distance_5.close()
        distance_5 = np.array(distance_dic_5['distance'])
        distance_idx_2 = grid_idx[np.where(distance >= 13.875)]
        distance_idx_3 = grid_idx[np.where(distance >= (2 ** 0.5) * 13.875)]
        distance_idx_4 = grid_idx[np.where(distance >= 2 * 13.875)]
        distance_idx_5 = grid_idx[np.where(distance >= (5 ** 0.5) * 13.875)]
        adjacency_1 = np.array(distance_1)
        adjacency_2 = distance_2[:, np.where((np.in1d(distance_2[0], distance_idx_2) == True) & (np.in1d(distance_2[1], distance_idx_2) == True))[0]]
        adjacency_3 = distance_3[:, np.where((np.in1d(distance_3[0], distance_idx_3) == True) & (np.in1d(distance_3[1], distance_idx_3) == True))[0]]
        adjacency_4 = distance_4[:, np.where((np.in1d(distance_4[0], distance_idx_4) == True) & (np.in1d(distance_4[1], distance_idx_4) == True))[0]]
        adjacency_5 = distance_5[:, np.where((np.in1d(distance_5[0], distance_idx_5) == True) & (np.in1d(distance_5[1], distance_idx_5) == True))[0]]
        edge_idx = np.concatenate([adjacency_1, adjacency_2, adjacency_3, adjacency_4, adjacency_5], axis = 1)
        edge_idx_start = edge_idx[0]
        edge_idx_start = edge_idx[:, np.where(np.isnan(sst_grid.flatten()[edge_idx_start]) == False)[0]]
        edge_idx_end = edge_idx_start[1]
        edge_idx = edge_idx_start[:, np.where(np.isnan(sst_grid.flatten()[edge_idx_end]) == False)[0]]
        lat_grid = np.arange(lat_min, lat_max + 1, 0.125)
        lon_grid = np.arange(np.min(lon), np.min(lon) + 40, 0.125)
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
        lat_grid = lat_grid.flatten()
        lon_grid = lon_grid.flatten()

        start_lon = lon_grid[edge_idx[0]]
        start_lat = lat_grid[edge_idx[0]]
        end_lon = lon_grid[edge_idx[1]]
        end_lat = lat_grid[edge_idx[1]]
        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
        ax.set_extent([np.min(lon), np.min(lon) + 32, lat_min, lat_max])
        for n in range(0, len(start_lon), 1):
            plt.plot([start_lon[n] - 180, end_lon[n] - 180], [start_lat[n], end_lat[n]], c = 'lightgrey', linewidth = 0.1)
        plt.pcolor(np.arange(np.min(lon), np.min(lon) + 40, 0.125), np.arange(lat_min, lat_max + 1, 0.125), sst_grid, transform = ccrs.PlateCarree(central_longitude = 0), cmap = cmaps.BlueDarkRed18)
        plt.savefig(plotdir + '/Edge.png', dpi = 1000, bbox_inches = 'tight')

        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        ax.set_extent([np.min(lon) + 5 + 0.0625, np.min(lon) + 7 - 0.0625, lat_min + 10 + 0.0625, lat_min + 12 - 0.0625])
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.3)
        ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        plt.scatter(lon, lat, color = 'grey', edgecolors = 'black', marker = 'o', s = 200, transform = ccrs.PlateCarree(), zorder = 10)
        for n in range(0, len(start_lon), 1):
            if (start_lon[n] == np.min(lon) + 6 + 0.125) and (start_lat[n] == lat_min + 11 + 0.125):
                plt.plot([start_lon[n] - 180, end_lon[n] - 180], [start_lat[n], end_lat[n]], c = rgb_to_hex('053,183,119'), linewidth = 5)
                plt.scatter(start_lon[n], start_lat[n], color = rgb_to_hex('053,183,119'), marker = 'o', s = 200, edgecolors = 'black', transform = ccrs.PlateCarree(), zorder = 20)
            if (start_lon[n] == np.min(lon) + 7 - 0.5) and (start_lat[n] == lat_min + 10 + 0.375):
                plt.plot([start_lon[n] - 180, end_lon[n] - 180], [start_lat[n], end_lat[n]], c = rgb_to_hex('075,101,175'), linewidth = 5)
                plt.scatter(start_lon[n], start_lat[n], color = rgb_to_hex('075,101,175'), marker = 'o', s = 200, edgecolors = 'black', transform = ccrs.PlateCarree(), zorder = 20)
            if (start_lon[n] == np.min(lon) + 5 + 0.5) and (start_lat[n] == lat_min + 12 - 0.5):
                plt.plot([start_lon[n] - 180, end_lon[n] - 180], [start_lat[n], end_lat[n]], c = rgb_to_hex('244,111,068'), linewidth = 5)
                plt.scatter(start_lon[n], start_lat[n], color = rgb_to_hex('244,111,068'), marker = 'o', s = 200, edgecolors = 'black', transform = ccrs.PlateCarree(), zorder = 20)
        plt.savefig(plotdir + '/Edge_index.png', dpi = 1000, bbox_inches = 'tight')


    def plot_input_sswsst(sswsst_data):
        sswsst = np.array(sswsst_data['sswsst'])
        lat = np.array(sswsst_data['lat'])
        for var_num in range(3):
            lat_idx = ((lat + 90) * 4).astype(int)
            lon_idx = np.array(sswsst_data['lon'])
            lon_idx[np.where(lon_idx < 0)] = 360 + lon_idx[np.where(lon_idx < 0)]
            lon_idx = (lon_idx * 4).astype(int)
            sswsst_domain = np.full([180 * 4 + 1, 360 * 4 + 1], np.nan)
            sswsst_domain[lat_idx, lon_idx] = sswsst[var_num]
            plt.figure(figsize = (8, 8))
            ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
            ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
            ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree(central_longitude = 180))
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            plt.pcolor(np.arange(0, 360.01, 0.25), np.arange(-90, 90.01, 0.25), sswsst_domain, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18)
            plt.savefig(plotdir + '/Input_Var{0}.png'.format(var_num + 1), dpi = 1000, bbox_inches = 'tight')

    def plot_input_sswsst(sswsst_data):
        sswsst = np.array(sswsst_data['sswsst'])
        lat = np.array(sswsst_data['lat'])
        lon = np.array(sswsst_data['lat'])
        # for var_num in range(3):
        #     lat_idx = ((lat + 90) * 4).astype(int)
        #     lon_idx = np.array(sswsst_data['lon'])
        #     lon_idx[np.where(lon_idx < 0)] = 360 + lon_idx[np.where(lon_idx < 0)]
        #     lon_idx = (lon_idx * 4).astype(int)
        #     sswsst_domain = np.full([180 * 4 + 1, 360 * 4 + 1], np.nan)
        #     sswsst_domain[lat_idx, lon_idx] = sswsst[var_num]
        #     plt.figure(figsize = (8, 8))
        #     ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        #     ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        #     ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree(central_longitude = 180))
        #     ax.xaxis.set_major_formatter(LongitudeFormatter())
        #     ax.yaxis.set_major_formatter(LatitudeFormatter())
        #     plt.pcolor(np.arange(0, 360.01, 0.25), np.arange(-90, 90.01, 0.25), sswsst_domain, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18)
        #     plt.savefig(plotdir + '/Input_Var{0}.png'.format(var_num + 1), dpi = 1000, bbox_inches = 'tight')
        domain = domain_list[1]
        lat_min = domain[0]
        lat_max = domain[1]
        sswsst = np.array(sswsst_data['sswsst'])
        lat = np.array(sswsst_data['lat'])
        lon = np.array(sswsst_data['lon'])
        slice_domain_idx = np.where((lat >= lat_min) & (lat < lat_max + 1))[0]
        sswsst = sswsst[:, slice_domain_idx]
        lat = lat[slice_domain_idx]
        lon = lon[slice_domain_idx]
        lat_idx = ((lat + 90) * 4).astype(int)
        lon[np.where(lon < 0)] = 360 + lon[np.where(lon < 0)]
        lon_idx = (lon * 4).astype(int)
        for var_num in range(3):
            sswsst_domain = np.full([180 * 4 + 1, 360 * 4 + 1], np.nan)
            if var_num == 0:
                sswsst_domain[lat_idx, lon_idx] = (sswsst[0] ** 2 + sswsst[1] ** 2) ** 0.5
            plt.figure(figsize = (8, 8))
            ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
            ax.set_extent([np.min(lon), np.min(lon) + 32, lat_min, lat_max])
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.3)
            ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            plt.pcolor(np.arange(-180, 180.01, 0.25), np.arange(-90, 90.01, 0.25), sswsst_domain, transform = ccrs.PlateCarree(central_longitude = 180), cmap = cmaps.BlueDarkRed18)
            plt.savefig(plotdir + '/SSWSST{0}_Domain.png'.format(var_num), dpi = 1000, bbox_inches = 'tight')

    def plot_output(sswsst_data):
        sswsst = np.array(sswsst_data['sswsst'])
        lat = np.array(sswsst_data['lat'])
        for var_num in range(3):
            lat_idx = ((lat + 90) * 4).astype(int)
            lon_idx = np.array(sswsst_data['lon'])
            lon_idx[np.where(lon_idx < 0)] = 360 + lon_idx[np.where(lon_idx < 0)]
            lon_idx = (lon_idx * 4).astype(int)
            sswsst_domain = np.full([180 * 4 + 1, 360 * 4 + 1], np.nan)
            sswsst_domain[lat_idx, lon_idx] = sswsst[var_num]
            plt.figure(figsize = (8, 8))
            ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
            ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
            ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree(central_longitude = 180))
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            plt.pcolor(np.arange(0, 360.01, 0.25), np.arange(-90, 90.01, 0.25), sswsst_domain, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18)
            plt.savefig(plotdir + '/Output_Var{0}.png'.format(var_num + 1), dpi = 1000, bbox_inches = 'tight')

    plotdir = abspath + '/Save/Plot/Flow'
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

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
    idx = 41
    bt_data = xr.open_dataset(loc_bt + '/{0}/FY3_{1}_bt_{2}'.format(val_list[idx][0].capitalize(), val_list[idx][0], val_list[idx][1]))
    sswsst_data = xr.open_dataset(loc_sswsst + '/{0}/ERA5_{1}_sswsst_{2}'.format(val_list[idx][0].capitalize(), val_list[idx][0], val_list[idx][1]))

    # plot_domain(bt_data)
    # plot_input_bt(bt_data)
    # plot_input_saa(bt_data)
    # plot_input_topography()
    # plot_edge(bt_data)
    plot_positional_encoding(bt_data)
    # plot_input_sswsst(sswsst_data)
    # plot_output(sswsst_data)

def edge(domain):
    plotdir = abspath + '/Save/Plot/Edge'
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    lat_min = int(domain.split(',')[0])
    lat_max = int(domain.split(',')[1])
    lon_len = int(domain.split(',')[2])
    lat_len = lat_max - lat_min
    lat_grid = np.arange(lat_min, lat_max, 0.125)
    lon_grid = np.arange(0, lon_len, 0.125)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    sst_gradient = []
    loc = abspath + '/DataSet/X/FY3'
    for orbit in ['asc', 'dsc']:
        file_list = loc + '/{0}'.format(orbit.capitalize())
        file_list = sorted([file_list + '/' + i for i in os.listdir(file_list)])
        file_list = file_list[::400]
        for file in file_list:
            if file.split('_')[1] == orbit:
                fy3 = xr.open_dataset(file)
                lat = np.array(fy3['lat'])
                lon = np.array(fy3['lon'])
                slice_bt = np.where((lat >= lat_min) & (lat < lat_max))[0]
                if len(slice_bt) >= 10:
                    bt = np.array(fy3['bt']).transpose(1, 0)
                    sst = (np.dot(bt, np.array([[0.82], [-0.47], [0.51], [-0.29], [0.06], [0.03], [-0.58], [0.27], [0.06], [-0.03]])) - 78.26 + 273.15)[:, 0]
                    sst = sst[slice_bt]
                    lat = lat[slice_bt]
                    lon = lon[slice_bt]
                    if np.max(lon) - np.min(lon) > 180:
                        lon_adjust_idx = np.where(lon < 0)
                        lon[lon_adjust_idx] = lon[lon_adjust_idx] + 360
                    lat = ((lat - np.min(lat)) * 8).astype(int)
                    lon = ((lon - np.min(lon)) * 8).astype(int)
                    if (np.max(lon) >= (lon_len * 8)):
                        slice_bt = np.where(lon < (lon_len * 8))
                        sst = sst[slice_bt]
                        lat = lat[slice_bt]
                        lon = lon[slice_bt]
                    sst_grid = np.full([lat_len * 8, lon_len * 8], np.nan)
                    sst_grid[lat, lon] = sst
                    dx, dy = mpcalc.lat_lon_grid_deltas(lon_grid, lat_grid)
                    sst_grid = np.array(mpcalc.gradient(sst_grid, deltas = (dy, dx)))
                    sst_grid = ((sst_grid[0] ** 2 + sst_grid[1] ** 2) ** 0.5) * 1000
                    sst_grid = sst_grid.flatten()
                    sst_grid = sst_grid[np.where(np.isnan(sst_grid) == False)]
                    sst_grid = sst_grid[np.where(sst_grid < 0.1)]
                    sst_gradient.append(sst_grid)
    sst_gradient = np.concatenate(sst_gradient).flatten()
    sst_gradient = sorted(sst_gradient)
    pdf, _ = np.histogram(sst_gradient, bins = 50, density = True)
    pdf = list(pdf)
    pdf.append(pdf[-1])
    plt.figure(figsize = (14, 10))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.bar(np.linspace(0, 0.1, 51), pdf, color = 'steelblue', label = 'Probability Distribution', width = 0.002, edgecolor = 'black')
    # ax1.plot(np.nan, np.nan, color = 'orange', linewidth = 5, label = 'd$_{\mathregular{n}}$=αg$_{\mathregular{n}}$+β, (α=-250, β=40)')
    ax1.plot(np.nan, np.nan, color = 'orange', linewidth = 5, label = 'D=αG+β, (α=-250, β=40)')
    ax2.plot(np.arange(0, 0.1, 0.001), (-250) * np.arange(0, 0.1, 0.001) + 40, color = 'orange', linewidth = 5)
    ax1.legend(loc = 'upper right', frameon = False, fontsize = 26)
    ax1.set_xlabel('SST Gradient °C/km', fontsize = 32)
    ax1.set_ylabel('SST Probability Density (%)', fontsize = 32)
    ax2.set_ylabel('Distance Threshold (km)', fontsize = 32)
    ax1.set_ylim(0, 40)
    ax2.set_ylim(10, 50)
    ax1.tick_params(labelsize = 28)
    ax2.tick_params(labelsize = 28)
    plt.tick_params(pad = 7, labelsize = 30)
    plt.savefig(plotdir + '/PDF_{}-{}.png'.format(lat_min, lat_max), dpi = 1000, bbox_inches = 'tight')

def conclusion():
    def plot_error_rmse():
        plotdir = plot_loc + '/Error'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        rmse_sws_ssw_list = [[] for _ in range(len(domain_list))]
        for date in date_list:
            sws_asc = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            era5_asc = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            sws_asc_ssws = (sws_asc[0] ** 2 + sws_asc[1] ** 2) ** 0.5
            era5_asc_ssws = (era5_asc[0] ** 2 + era5_asc[1] ** 2) ** 0.5
            for domain_num in range(len(domain_list)):
                lat_min = domain_list[domain_num][0]
                lat_max = domain_list[domain_num][1]
                rmse = np.nanmean((np.array(sws_asc_ssws.loc[lat_min:lat_max]) - np.array(era5_asc_ssws.loc[lat_min:lat_max])) ** 2) ** 0.5
                rmse_sws_ssw_list[domain_num].append(rmse)
        rmse_sws_ssw_list = np.array(rmse_sws_ssw_list)
        rmse_sws_sst_list = [[] for _ in range(len(domain_list))]
        for date in date_list:
            sws_asc = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            era5_asc = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            sws_asc_sst = sws_asc[2]
            era5_asc_sst = era5_asc[2]
            for domain_num in range(len(domain_list)):
                lat_min = domain_list[domain_num][0]
                lat_max = domain_list[domain_num][1]
                rmse = np.nanmean((np.array(sws_asc_sst.loc[lat_min:lat_max]) - np.array(era5_asc_sst.loc[lat_min:lat_max])) ** 2) ** 0.5
                rmse_sws_sst_list[domain_num].append(rmse)
        rmse_sws_sst_list = np.array(rmse_sws_sst_list)
        rmse_sws_sswd_list = [[] for _ in range(len(domain_list))]
        for date in date_list:
            sws = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            era5 = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            era5_ssws = (era5[0] ** 2 + era5[1] ** 2) ** 0.5
            for domain_num in range(len(domain_list)):
                lat_min = domain_list[domain_num][0]
                lat_max = domain_list[domain_num][1]
                sws_sswd = 180 + np.arctan2(np.array(sws[0].loc[lat_min:lat_max]), np.array(sws[1].loc[lat_min:lat_max])) * (180 / np.pi)
                era5_sswd = 180 + np.arctan2(np.array(era5[0].loc[lat_min:lat_max]), np.array(era5[1].loc[lat_min:lat_max])) * (180 / np.pi)
                sws_sswd = sws_sswd[np.where(np.array(era5_ssws.loc[lat_min:lat_max]) >= 4)]
                era5_sswd = era5_sswd[np.where(np.array(era5_ssws.loc[lat_min:lat_max]) >= 4)]
                error_sswd = np.abs(sws_sswd - era5_sswd)
                adjust_error_sswd_idx = np.where(error_sswd >= 180)
                error_sswd[adjust_error_sswd_idx] = 360 - error_sswd[adjust_error_sswd_idx]
                rmse = np.nanmean(error_sswd ** 2) ** 0.5
                rmse_sws_sswd_list[domain_num].append(rmse)
        rmse_sws_sswd_list = np.array(rmse_sws_sswd_list)
        errorbar_ssw = []
        errorbar_sst = []
        errorbar_sswd = []
        for domain_num in range(len(domain_list)):
            lower, upper = stats.norm.interval(0.95, loc = np.mean(rmse_sws_ssw_list[domain_num]), scale = stats.sem(rmse_sws_ssw_list[domain_num]))
            errorbar_ssw.append((upper - lower) / 2)
            lower, upper = stats.norm.interval(0.95, loc = np.mean(rmse_sws_sst_list[domain_num]), scale = stats.sem(rmse_sws_sst_list[domain_num]))
            errorbar_sst.append((upper - lower) / 2)
            lower, upper = stats.norm.interval(0.95, loc = np.mean(rmse_sws_sswd_list[domain_num]), scale = stats.sem(rmse_sws_sswd_list[domain_num]))
            errorbar_sswd.append((upper - lower) / 2)
        print(np.mean(rmse_sws_ssw_list, axis = 1))
        print(np.mean(rmse_sws_sst_list, axis = 1))
        print(np.mean(rmse_sws_sswd_list, axis = 1))
        # plt.figure(figsize = (10, 8))
        # ax1 = plt.gca()
        # ax1.barh(np.arange(len(domain_list)) + 0.25, np.mean(rmse_sws_ssw_list, axis = 1), height = 0.25, xerr = errorbar_ssw, error_kw = {'ecolor':'black', 'elinewidth':1.5, 'capsize':3, 'capthick':1.5}, \
        #          color = '#66b0d7', label = None, edgecolor = 'black', zorder = 10)
        # ax1.barh(np.arange(len(domain_list)), np.mean(rmse_sws_sst_list, axis = 1), height = 0.25, xerr = errorbar_sst, error_kw = {'ecolor':'black', 'elinewidth':1.5, 'capsize':3, 'capthick':1.5}, \
        #          color = '#f1b74e', label = None, edgecolor = 'black', zorder = 20)
        # ax2 = ax1.twiny()
        # ax2.barh(np.arange(len(domain_list)) - 0.25, np.mean(rmse_sws_sswd_list, axis = 1), height = 0.25, xerr = errorbar_sswd, error_kw = {'ecolor':'black', 'elinewidth':1.5, 'capsize':3, 'capthick':1.5}, \
        #          color = '#e38d8c', label = None, edgecolor = 'black', zorder = 30)
        # ax1.set_xlabel('SSWS RMSE (m/s) & SST RMSE (K)', fontsize = 30)
        # ax2.set_xlabel('SSWD RMSE (Degree)', labelpad = 10, fontsize = 30)
        # ax1.set_ylabel('Domain', fontsize = 30)
        # ax1.set_xlim(0, 1.4)
        # ax2.set_xlim(0, 21)
        # ax1.set_ylim(-0.7, 8.7)
        # ax2.xaxis.set_major_locator(MultipleLocator(3))
        # ax1.set_yticks(np.arange(len(domain_list)), labels = ['9 & 10', '8', '7', '6', '5', '4', '3', '1 & 2', 'Global'])
        # ax1.grid(axis = 'x', color = 'grey', alpha = 0.3, linestyle = '--')
        # ax1.tick_params(labelsize = 20)
        # ax2.tick_params(labelsize = 20)
        # plt.title('(a)', loc = 'left', fontsize = 35, weight = 'bold')
        # plt.savefig(plotdir + '/(a) RMSE.png', dpi = 1000, bbox_inches = 'tight')

    def plot_error_corr():
        plotdir = plot_loc + '/Error'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        corr_sws_ssw_list = [[] for _ in range(len(domain_list))]
        for date in date_list:
            sws_asc = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            era5_asc = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            sws_asc_ssws = (sws_asc[0] ** 2 + sws_asc[1] ** 2) ** 0.5
            era5_asc_ssws = (era5_asc[0] ** 2 + era5_asc[1] ** 2) ** 0.5
            for domain_num in range(len(domain_list)):
                lat_min = domain_list[domain_num][0]
                lat_max = domain_list[domain_num][1]
                sws_asc_ssws_part = np.array(sws_asc_ssws.loc[lat_min:lat_max]).flatten()
                era5_asc_ssws_part = np.array(era5_asc_ssws.loc[lat_min:lat_max]).flatten()
                slice_nan = np.where(np.isnan(sws_asc_ssws_part) == False)[0]
                sws_asc_ssws_part = sws_asc_ssws_part[slice_nan]
                era5_asc_ssws_part = era5_asc_ssws_part[slice_nan]
                slice_nan = np.where(np.isnan(era5_asc_ssws_part) == False)[0]
                sws_asc_ssws_part = sws_asc_ssws_part[slice_nan]
                era5_asc_ssws_part = era5_asc_ssws_part[slice_nan]
                corr = np.corrcoef(sws_asc_ssws_part, era5_asc_ssws_part)[0, 1]
                corr_sws_ssw_list[domain_num].append(corr)
        corr_sws_ssw_list = np.array(corr_sws_ssw_list)
        corr_sws_sst_list = [[] for _ in range(len(domain_list))]
        for date in date_list:
            sws_asc = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            era5_asc = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            sws_asc_sst = sws_asc[2]
            era5_asc_sst = era5_asc[2]
            for domain_num in range(len(domain_list)):
                lat_min = domain_list[domain_num][0]
                lat_max = domain_list[domain_num][1]
                sws_asc_sst_part = np.array(sws_asc_sst.loc[lat_min:lat_max]).flatten()
                era5_asc_sst_part = np.array(era5_asc_sst.loc[lat_min:lat_max]).flatten()
                slice_nan = np.where(np.isnan(sws_asc_sst_part) == False)[0]
                sws_asc_sst_part = sws_asc_sst_part[slice_nan]
                era5_asc_sst_part = era5_asc_sst_part[slice_nan]
                slice_nan = np.where(np.isnan(era5_asc_sst_part) == False)[0]
                sws_asc_sst_part = sws_asc_sst_part[slice_nan]
                era5_asc_sst_part = era5_asc_sst_part[slice_nan]
                corr = np.corrcoef(sws_asc_sst_part, era5_asc_sst_part)[0, 1]
                corr_sws_sst_list[domain_num].append(corr)
        corr_sws_sst_list = np.array(corr_sws_sst_list)
        corr_sws_sswd_list = [[] for _ in range(len(domain_list))]
        for date in date_list:
            sws = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            era5 = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_asc_sswsst_{0}.nc'.format(date))['sswsst'][0]
            era5_ssws = (era5[0] ** 2 + era5[1] ** 2) ** 0.5
            for domain_num in range(len(domain_list)):
                lat_min = domain_list[domain_num][0]
                lat_max = domain_list[domain_num][1]
                sws_sswd = 180 + np.arctan2(np.array(sws[0].loc[lat_min:lat_max]), np.array(sws[1].loc[lat_min:lat_max])) * (180 / np.pi)
                era5_sswd = 180 + np.arctan2(np.array(era5[0].loc[lat_min:lat_max]), np.array(era5[1].loc[lat_min:lat_max])) * (180 / np.pi)
                sws_sswd = sws_sswd[np.where(np.array(era5_ssws.loc[lat_min:lat_max]) >= 4)]
                era5_sswd = era5_sswd[np.where(np.array(era5_ssws.loc[lat_min:lat_max]) >= 4)]
                error_sswd = np.abs(sws_sswd - era5_sswd)
                adjust_error_sswd_idx = np.where(error_sswd < 180)
                sws_sswd_part = np.array(sws_sswd[adjust_error_sswd_idx]).flatten()
                era5_sswd_part = np.array(era5_sswd[adjust_error_sswd_idx]).flatten()
                slice_nan = np.where(np.isnan(sws_sswd_part) == False)[0]
                sws_sswd_part = sws_sswd_part[slice_nan]
                era5_sswd_part = era5_sswd_part[slice_nan]
                slice_nan = np.where(np.isnan(era5_sswd_part) == False)[0]
                sws_sswd_part = sws_sswd_part[slice_nan]
                era5_sswd_part = era5_sswd_part[slice_nan]
                corr = np.corrcoef(sws_sswd_part, era5_sswd_part)[0, 1]
                corr_sws_sswd_list[domain_num].append(corr)
        corr_sws_sswd_list = np.array(corr_sws_sswd_list)
        errorbar_ssw = []
        errorbar_sst = []
        errorbar_sswd = []
        for domain_num in range(len(domain_list)):
            lower, upper = stats.norm.interval(0.95, loc = np.mean(corr_sws_ssw_list[domain_num]), scale = stats.sem(corr_sws_ssw_list[domain_num]))
            errorbar_ssw.append((upper - lower) / 2)
            lower, upper = stats.norm.interval(0.95, loc = np.mean(corr_sws_sst_list[domain_num]), scale = stats.sem(corr_sws_sst_list[domain_num]))
            errorbar_sst.append((upper - lower) / 2)
            lower, upper = stats.norm.interval(0.95, loc = np.mean(corr_sws_sswd_list[domain_num]), scale = stats.sem(corr_sws_sswd_list[domain_num]))
            errorbar_sswd.append((upper - lower) / 2)
        plt.figure(figsize = (10, 8))
        ax1 = plt.gca()
        ax1.barh(np.arange(len(domain_list)) + 0.25, np.mean(corr_sws_ssw_list, axis = 1), height = 0.25, xerr = errorbar_ssw, error_kw = {'ecolor':'black', 'elinewidth':1.5, 'capsize':3, 'capthick':1.5}, \
                 color = '#66b0d7', label = None, edgecolor = 'black', zorder = 10)
        ax1.barh(np.arange(len(domain_list)), np.mean(corr_sws_sst_list, axis = 1), height = 0.25, xerr = errorbar_sst, error_kw = {'ecolor':'black', 'elinewidth':1.5, 'capsize':3, 'capthick':1.5}, \
                 color = '#f1b74e', label = None, edgecolor = 'black', zorder = 20)
        ax2 = ax1.twiny()
        ax2.barh(np.arange(len(domain_list)) - 0.25, np.mean(corr_sws_sswd_list, axis = 1), height = 0.25, xerr = errorbar_sswd, error_kw = {'ecolor':'black', 'elinewidth':1.5, 'capsize':3, 'capthick':1.5}, \
                 color = '#e38d8c', label = None, edgecolor = 'black', zorder = 20)
        ax1.set_xlabel('Correlation coefficient', fontsize = 30)
        ax2.set_xlabel(' ', labelpad = 10, fontsize = 30)
        ax1.set_ylabel('Domain', fontsize = 30)
        ax1.set_xlim(0.93, 1)
        ax2.set_xlim(0.93, 1)
        ax1.set_ylim(-0.7, 8.7)
        ax2.set_xticks(np.arange(0.93, 1.001, 0.01), labels = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
        ax1.set_yticks(np.arange(len(domain_list)), labels = ['9 & 10', '8', '7', '6', '5', '4', '3', '1 & 2', 'Global'])
        ax1.grid(axis = 'x', color = 'grey', alpha = 0.3, linestyle = '--')
        ax1.tick_params(labelsize = 20)
        ax2.tick_params(labelsize = 20)
        plt.title('(b)', loc = 'left', fontsize = 35, weight = 'bold')
        plt.savefig(plotdir + '/(b) Correlation coefficient.png', dpi = 1000, bbox_inches = 'tight')
        plt.figure(figsize = (10, 8))
        ax1 = plt.gca()
        ax1.barh(np.arange(len(domain_list)) + 0.2, np.mean(corr_sws_ssw_list, axis = 1) / 100, height = 0.4, color = '#66b0d7', label = 'SSWS retrieved by DGPRN', edgecolor = 'black', zorder = 10)
        ax1.barh(np.arange(len(domain_list)) - 0.2, np.mean(corr_sws_sst_list, axis = 1) / 100, height = 0.4, color = '#f1b74e', label = 'SST retrieved by DGPRN', edgecolor = 'black', zorder = 20)
        ax1.barh(np.arange(len(domain_list)) - 0.2, np.mean(corr_sws_sswd_list, axis = 1) / 100, height = 0.4, color = '#e38d8c', label = 'SSWD retrieved by DGPRN', edgecolor = 'black', zorder = 20)
        ax2 = ax1.twiny()
        ax1.set_xlabel('Correlation coefficient', fontsize = 30)
        ax2.set_xlabel(' ', labelpad = 10, fontsize = 30)
        ax1.set_ylabel('Domain', fontsize = 30)
        ax1.set_xlim(0.93, 1)
        ax2.set_xlim(0.93, 1)
        ax1.set_ylim(-0.7, 8.7)
        ax2.set_xticks(np.arange(0.93, 1.001, 0.01), labels = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
        ax1.set_yticks(np.arange(len(domain_list)), labels = ['9 & 10', '8', '7', '6', '5', '4', '3', '1 & 2', 'Global'])
        ax1.tick_params(labelsize = 20)
        ax2.tick_params(labelsize = 20)
        # ax1.legend(frameon = False, fontsize = 30)
        legend = ax1.legend(frameon = True, framealpha = 1, ncol = 3, fontsize = 30)
        legend.set_zorder(20)
        plt.title('(c)', loc = 'left', fontsize = 35, weight = 'bold')
        plt.savefig(plotdir + '/(c) Legend.png', dpi = 1000, bbox_inches = 'tight')

    def plot_rmse_spatial():
        plotdir = plot_loc + '/RMSE'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        rmse_sws_ssw_list = []
        for date in date_list:
            for orbit in ['asc']:
                sws = xr.open_dataset(abspath + '/Note/Output/SWS2/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
                era5 = xr.open_dataset(abspath + '/Note/Output/ERA52/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
                sws_ssws = (sws[0] ** 2 + sws[1] ** 2) ** 0.5
                era5_ssws = (era5[0] ** 2 + era5[1] ** 2) ** 0.5
                rmse_sws_ssw_list.append(np.array(sws_ssws.loc[-60:60]) - np.array(era5_ssws.loc[-60:60]))
        rmse_sws_ssw_list = np.array(rmse_sws_ssw_list)
        rmse_sws_ssw_list = np.nanmean(rmse_sws_ssw_list ** 2, axis = 0) ** 0.5
        lon = np.array(sws_ssws.loc[-60:60]['lon'])
        lat = np.array(sws_ssws.loc[-60:60]['lat'])
        rmse_sws_ssw_list[:, 180 * 4] = rmse_sws_ssw_list[:, 180 * 4 + 1]
        rmse_sws_ssw_list[:, 180 * 4 - 1] = rmse_sws_ssw_list[:, 180 * 4 + 1]
        rmse_sws_ssw_list[:, 180 * 4 - 2] = rmse_sws_ssw_list[:, 179 * 4 - 1]
        rmse_sws_ssw_list[:, 180 * 4 - 3] = rmse_sws_ssw_list[:, 179 * 4 - 1]
        rmse_sws_ssw_list[:, 179 * 4] = rmse_sws_ssw_list[:, 179 * 4 - 1]
        rmse_sws_ssw_list, lon = add_cyclic_point(rmse_sws_ssw_list, coord = lon)
        rmse_sws_ssw_list_temp = rmse_sws_ssw_list.copy()
        window = np.ones(int(5)) / float(5)
        for lat_num in range(len(lat)):
            rmse_sws_ssw_list_temp[lat_num] = np.convolve(rmse_sws_ssw_list[lat_num], window, 'same')
        for lon_convolve in [180]:
            part = rmse_sws_ssw_list[:, int((lon_convolve - 2) * 4):int((lon_convolve + 2) * 4)]
            part_convolve = rmse_sws_ssw_list_temp[:, int((lon_convolve - 2) * 4):int((lon_convolve + 2) * 4)]
            for lat_num in range(part.shape[0]):
                for lon_num in range(part.shape[1]):
                    if np.isnan(part_convolve[lat_num, lon_num]) == False:
                        part[lat_num, lon_num] = part_convolve[lat_num, lon_num]
            rmse_sws_ssw_list[:, int((lon_convolve - 2) * 4):int((lon_convolve + 2) * 4)] = part
        fig = plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5)
        ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        ax.set_extent([-180, 180, -60, 60], crs = ccrs.PlateCarree(central_longitude = 180))
        ax.set_xticks(np.arange(-180, 180.1, 60))
        ax.set_yticks(np.arange(-60, 60.1, 20))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis = 'both', which = 'major', labelsize = 15, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
        pcolor = plt.pcolor(lon - 180, lat, rmse_sws_ssw_list, cmap = 'rainbow', vmin = 0, vmax = 2)
        colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05)
        colorbar.set_ticks(np.arange(0, 2.01, 0.4))
        colorbar.set_label(label = 'SSWS RMSE (m/s)', fontsize = 20)
        colorbar.ax.tick_params(labelsize = 20)
        plt.title('SSWS retrieved by DGPRN', loc = 'center', fontsize = 20)
        plt.title('({0})'.format(order[0]), loc = 'left', fontsize = 35, weight = 'bold')
        plt.savefig(plotdir + '/RMSE_SWS.png', dpi = 1000, bbox_inches = 'tight')
        rmse_sws_sst_list = []
        for date in date_list:
            for orbit in ['asc']:
                sws = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
                era5 = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
                sws_sst = sws[2]
                era5_sst = era5[2]
                rmse_sws_sst_list.append(np.array(sws_sst.loc[-60:60]) - np.array(era5_sst.loc[-60:60]))
        rmse_sws_sst_list = np.array(rmse_sws_sst_list)
        rmse_sws_sst_list = np.nanmean(rmse_sws_sst_list ** 2, axis = 0) ** 0.5
        lon = np.array(sws_sst.loc[-60:60]['lon'])
        lat = np.array(sws_sst.loc[-60:60]['lat'])
        rmse_sws_sst_list[:, 180 * 4] = rmse_sws_sst_list[:, 180 * 4 + 1]
        rmse_sws_sst_list[:, 180 * 4 - 1] = rmse_sws_sst_list[:, 180 * 4 + 1]
        rmse_sws_sst_list[:, 180 * 4 - 2] = rmse_sws_sst_list[:, 179 * 4 - 1]
        rmse_sws_sst_list[:, 180 * 4 - 3] = rmse_sws_sst_list[:, 179 * 4 - 1]
        rmse_sws_sst_list[:, 179 * 4] = rmse_sws_sst_list[:, 179 * 4 - 1]
        for _ in range(3):
            rmse_sws_sst_list_temp = rmse_sws_sst_list.copy()
            window = np.ones(int(5)) / float(5)
            for lon_num in range(len(lon)):
                rmse_sws_sst_list_temp[:, lon_num] = np.convolve(rmse_sws_sst_list[:, lon_num], window, 'same')
            for lat_convolve in [15, 30, 45, 60, 75, 90]:
                part = rmse_sws_sst_list[int((lat_convolve - 2) * 4):int((lat_convolve + 2) * 4)]
                part_convolve = rmse_sws_sst_list_temp[int((lat_convolve - 2) * 4):int((lat_convolve + 2) * 4)]
                for lat_num in range(part.shape[0]):
                    for lon_num in range(part.shape[1]):
                        if np.isnan(part_convolve[lat_num, lon_num]) == False:
                            part[lat_num, lon_num] = part_convolve[lat_num, lon_num]
                rmse_sws_sst_list[int((lat_convolve - 2) * 4):int((lat_convolve + 2) * 4)] = part
        rmse_sws_sst_list_temp = rmse_sws_sst_list.copy()
        window = np.ones(int(5)) / float(5)
        for lat_num in range(len(lat)):
            rmse_sws_sst_list_temp[lat_num] = np.convolve(rmse_sws_sst_list[lat_num], window, 'same')
        for lon_convolve in [180]:
            part = rmse_sws_sst_list[:, int((lon_convolve - 2) * 4):int((lon_convolve + 2) * 4)]
            part_convolve = rmse_sws_sst_list_temp[:, int((lon_convolve - 2) * 4):int((lon_convolve + 2) * 4)]
            for lat_num in range(part.shape[0]):
                for lon_num in range(part.shape[1]):
                    if np.isnan(part_convolve[lat_num, lon_num]) == False:
                        part[lat_num, lon_num] = part_convolve[lat_num, lon_num]
            rmse_sws_sst_list[:, int((lon_convolve - 2) * 4):int((lon_convolve + 2) * 4)] = part
        rmse_sws_sst_list, lon = add_cyclic_point(rmse_sws_sst_list, coord = lon)
        fig = plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5)
        ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        ax.set_extent([-180, 180, -60, 60], crs = ccrs.PlateCarree(central_longitude = 180))
        ax.set_xticks(np.arange(-180, 180.1, 60))
        ax.set_yticks(np.arange(-60, 60.1, 20))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis = 'both', which = 'major', labelsize = 15, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
        pcolor = plt.pcolor(lon - 180, lat, rmse_sws_sst_list, cmap = 'rainbow', vmin = 0, vmax = 2)
        colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05)
        colorbar.set_ticks(np.arange(0, 2.01, 0.4))
        colorbar.set_label(label = 'SST RMSE (K)', fontsize = 20)
        colorbar.ax.tick_params(labelsize = 20)
        plt.title('SST retrieved by DGPRN', loc = 'center', fontsize = 20)
        plt.title('({0})'.format(order[1]), loc = 'left', fontsize = 35, weight = 'bold')
        plt.savefig(plotdir + '/RMSE_SST.png', dpi = 1000, bbox_inches = 'tight')

    def plot_rmse_tao_ssw(order_list):
        plotdir = plot_loc + '/RMSE_TAO'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        tao_loc = abspath + '/DataSet/Z/TAO/SSW'
        tao_path_list = sorted([tao_loc + '/' + i for i in os.listdir(tao_loc)])
        rmse_list = [[] for _ in range(4)]
        for tao_path in tao_path_list:
            print(tao_path)
            tao = xr.open_dataset(tao_path)
            height = int(tao['height'])
            lat = int(tao['lat'])
            lon = int(tao['lon'])
            tao = tao['tao']
            if lon < 0:
                lon_adjust = 360 + lon
            else:
                lon_adjust = lon
            date_list = list(pd.to_datetime(tao['time']).strftime('%Y.%m.%d'))
            tao = (np.log(10 / 1.52e-4) / np.log(height / 1.52e-4)) * np.array(tao)
            # tao = 8.87403 * (np.array(tao) / (np.log(4 / 0.0016)))
            slice_nan = np.where(np.isnan(tao) == False)
            tao = tao[slice_nan]
            date_list = list(np.array(date_list)[slice_nan])
            if (len(date_list) > 0) and (lon_adjust != 180):
                data_list = [[] for _ in range(4)]
                for date_num in range(len(date_list)):
                    fy3_path = abspath + '/DataSet/Z/FY3D/SSW/FY3D_MWRIX_GBAL_L2_SWS_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date_list[date_num].replace('.', ''))
                    if os.path.exists(fy3_path) == True:
                        orbit = tao_path.split('/')[-1].split('_')[1]
                        sws_asc = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_{0}_sswsst_{1}.nc'.format(orbit, date_list[date_num]))['sswsst'][0]
                        era5_asc = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date_list[date_num]))['sswsst'][0]
                        if orbit == 'asc':
                            fy3_asc = np.array(nc.Dataset(fy3_path)['SWS_Ascending'])
                        elif orbit == 'dsc':
                            fy3_asc = np.array(nc.Dataset(fy3_path)['SWS_Descending'])
                        fy3_asc[np.where(fy3_asc == -9999.)] = np.nan
                        fy3_asc[np.where(fy3_asc == 110.0)] = np.nan
                        sws_asc_ssws = (sws_asc[0] ** 2 + sws_asc[1] ** 2) ** 0.5
                        era5_asc_ssws = (era5_asc[0] ** 2 + era5_asc[1] ** 2) ** 0.5
                        if (np.array(fy3_asc[(90 - lat) * 4, (180 + lon) * 4]) <= 15) and (np.array(fy3_asc[(90 - lat) * 4, (180 + lon) * 4]) > 0):
                            data_list[0].append(np.array(sws_asc_ssws.loc[lat, lon_adjust]))
                            data_list[1].append(np.array(era5_asc_ssws.loc[lat, lon_adjust]))
                            data_list[2].append(np.array(fy3_asc[(90 - lat) * 4, (180 + lon) * 4]))
                            data_list[3].append(np.array(tao[date_num]))
                data_list = np.array(data_list)
                if len(np.where(np.isnan(data_list) == False)[0]) != 0:
                    rmse1 = np.nanmean((data_list[3] - data_list[1]) ** 2) ** 0.5
                    rmse2 = np.nanmean((data_list[3] - data_list[0]) ** 2) ** 0.5
                    if rmse1 < 1.25 and rmse2 < 1.25:
                        rmse_list[0].append(data_list[0])
                        rmse_list[1].append(data_list[1])
                        rmse_list[2].append(data_list[2])
                        rmse_list[3].append(data_list[3])
                        print(np.nanmean((data_list[3] - data_list[0]) ** 2) ** 0.5)
                        print(np.nanmean((data_list[3] - data_list[1]) ** 2) ** 0.5)
                        print(np.nanmean((data_list[3] - data_list[2]) ** 2) ** 0.5)
                    # slice_nan = np.where(np.isnan(data_list[0]) == False)
                    # plt.plot(np.arange(len(data_list[2][slice_nan])), data_list[0][slice_nan])
                    # plt.plot(np.arange(len(data_list[2][slice_nan])), data_list[1][slice_nan])
                    # plt.plot(np.arange(len(data_list[2][slice_nan])), data_list[2][slice_nan])
                    # plt.plot(np.arange(len(data_list[2][slice_nan])), data_list[3][slice_nan])
                    # print(data_list[3][slice_nan])
                    # plt.savefig('/iapdisk2/Python/SWS/Save/Plot/Conclusion/RMSE_TAO/a.png')
                    # aaa
        ylabel_list = ['DGPRN', 'ERA5', 'FY3D']
        slice_nan = np.where((np.isnan(np.concatenate(rmse_list[0])) == False) & (np.isnan(np.concatenate(rmse_list[1])) == False) & (np.isnan(np.concatenate(rmse_list[2])) == False))[0]
        for plt_num in range(3):
            plt.figure(figsize = (12, 8))
            rmse = np.mean((np.concatenate(rmse_list[3])[slice_nan] - np.concatenate(rmse_list[plt_num])[slice_nan]) ** 2) ** 0.5
            corr = np.corrcoef(np.concatenate(rmse_list[3])[slice_nan], np.concatenate(rmse_list[plt_num])[slice_nan])[0, 1]
            plt.hist2d(np.concatenate(rmse_list[3]), np.concatenate(rmse_list[plt_num]), bins = np.arange(0, 15, 0.2), cmin = 1, norm = LogNorm(), cmap = cmaps.MPL_RdYlBu_r)
            # text_sample_num = plt.text(x = 1, y = 13.7, s = 'Sample Number = {0}'.format(len(np.concatenate(rmse_list[3])[slice_nan])), fontsize = 20)
            # text_rmse = plt.text(x = 1, y = 13, s = 'RMSE ={0} m/s'.format(format(rmse, '.4f')), fontsize = 20)
            text_pccs = plt.text(x = 1, y = 12.3, s = 'Sample Number = {0}\nRMSE ={1} m/s\nCorr. ={2}'.format(len(np.concatenate(rmse_list[3])[slice_nan]), format(rmse, '.4f'), format(corr, '.4f')), fontsize = 20)
            # text_sample_num.set_bbox(dict(facecolor = 'white', alpha = 1, edgecolor = 'white'))
            # text_rmse.set_bbox(dict(facecolor = 'white', alpha = 1, edgecolor = 'white'))
            text_pccs.set_bbox(dict(facecolor = 'white', alpha = 1, edgecolor = 'white'))
            colorbar = plt.colorbar()
            colorbar.ax.tick_params(labelsize = 20)
            plt.grid(color = 'grey', alpha = 0.3, linestyle = '--')
            plt.plot([0, 15], [0, 15], c = 'black', linestyle = '--', linewidth = 3)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(3))
            ax.yaxis.set_major_locator(MultipleLocator(3))
            ax.set_aspect(1)
            plt.xlim(0, 15)
            plt.ylim(0, 15)
            plt.clim(1, 100)
            plt.tick_params(labelsize = 20)
            plt.xlabel('TAO SSWS (m/s)', fontsize = 25)
            plt.ylabel('{0} SSWS (m/s)'.format(ylabel_list[plt_num]), fontsize = 25)
            plt.title('({0})'.format(order[order_list[plt_num]]), loc = 'left', fontsize = 35, weight = 'bold')
            plt.savefig(plotdir + '/TAO_{0}_SSW'.format(ylabel_list[plt_num]), dpi = 1000, bbox_inches = 'tight')

    def plot_rmse_tao_sst(order_list):
        plotdir = plot_loc + '/RMSE_TAO'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        tao_loc = abspath + '/DataSet/Z/TAO/SST'
        tao_path_list = sorted([tao_loc + '/' + i for i in os.listdir(tao_loc)])
        rmse_list = [[] for _ in range(4)]
        for tao_path in tao_path_list:
            print(tao_path)
            tao = xr.open_dataset(tao_path)
            lat = int(tao['lat'])
            lon = int(tao['lon'])
            tao = tao['tao']
            if lon < 0:
                lon_adjust = 360 + lon
            else:
                lon_adjust = lon
            date_list = list(pd.to_datetime(tao['time']).strftime('%Y.%m.%d'))
            slice_nan = np.where(np.isnan(tao) == False)
            tao = tao[slice_nan]
            date_list = list(np.array(date_list)[slice_nan])
            if (len(date_list) > 0) and (lon_adjust != 180):
                data_list = [[] for _ in range(4)]
                for date_num in range(len(date_list)):
                    fy3_path = abspath + '/DataSet/Z/FY3D/SST/FY3D_MWRIX_GBAL_L2_SST_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date_list[date_num].replace('.', ''))
                    if os.path.exists(fy3_path) == True:
                        orbit = tao_path.split('/')[-1].split('_')[1]
                        sws_asc = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_{0}_sswsst_{1}.nc'.format(orbit, date_list[date_num]))['sswsst'][0]
                        era5_asc = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date_list[date_num]))['sswsst'][0]
                        if orbit == 'asc':
                            fy3_sst = np.array(nc.Dataset(fy3_path)['SST_Ascending']) / 100
                        elif orbit == 'dsc':
                            fy3_sst = np.array(nc.Dataset(fy3_path)['SST_Descending']) / 100
                        fy3_sst[np.where(fy3_sst == -99.99)] = np.nan
                        sws_asc_sst = sws_asc[2]
                        era5_asc_sst = era5_asc[2]
                        data_list[0].append(np.array(sws_asc_sst.loc[lat, lon_adjust]))
                        data_list[1].append(np.array(era5_asc_sst.loc[lat, lon_adjust]))
                        data_list[2].append(np.array(fy3_sst[(90 - lat) * 4, (180 + lon) * 4]))
                        data_list[3].append(np.array(tao[date_num]) + 273.15)
                data_list = np.array(data_list)
                if len(np.where(np.isnan(data_list) == False)[0]) != 0:
                    rmse_list[0].append(data_list[0])
                    rmse_list[1].append(data_list[1])
                    rmse_list[2].append(data_list[2])
                    rmse_list[3].append(data_list[3])
                    print(np.nanmean((data_list[3] - data_list[0]) ** 2) ** 0.5)
                    print(np.nanmean((data_list[3] - data_list[1]) ** 2) ** 0.5)
                    print(np.nanmean((data_list[3] - data_list[2]) ** 2) ** 0.5)
                    # slice_nan = np.where(np.isnan(data_list[0]) == False)
                    # plt.plot(np.arange(len(data_list[2][slice_nan])), data_list[0][slice_nan])
                    # plt.plot(np.arange(len(data_list[2][slice_nan])), data_list[1][slice_nan])
                    # plt.plot(np.arange(len(data_list[2][slice_nan])), data_list[2][slice_nan])
                    # plt.plot(np.arange(len(data_list[2][slice_nan])), data_list[3][slice_nan])
                    # print(data_list[3][slice_nan])
                    # plt.savefig('/iapdisk2/Python/SWS/Save/Plot/Conclusion/RMSE_TAO/a.png')
                    # aaa
        ylabel_list = ['DGPRN', 'ERA5', 'FY3D']
        slice_nan = np.where((np.isnan(np.concatenate(rmse_list[0])) == False) & (np.isnan(np.concatenate(rmse_list[1])) == False) & (np.isnan(np.concatenate(rmse_list[2])) == False))[0]
        for plt_num in range(3):
            plt.figure(figsize = (12, 8))
            rmse = np.mean((np.concatenate(rmse_list[3])[slice_nan] - np.concatenate(rmse_list[plt_num])[slice_nan]) ** 2) ** 0.5
            corr = np.corrcoef(np.concatenate(rmse_list[3])[slice_nan], np.concatenate(rmse_list[plt_num])[slice_nan])[0, 1]
            plt.hist2d(np.concatenate(rmse_list[3]), np.concatenate(rmse_list[plt_num]), bins = np.arange(292, 307, 0.2), cmin = 1, norm = LogNorm(), cmap = cmaps.MPL_RdYlBu_r)
            # text_sample_num = plt.text(x = 293, y = 305.7, s = 'Sample Number = {0}'.format(len(np.concatenate(rmse_list[3])[slice_nan])), fontsize = 20)
            # text_rmse = plt.text(x = 293, y = 305, s = 'RMSE ={0} K'.format(format(rmse, '.4f')), fontsize = 20)
            text_pccs = plt.text(x = 293, y = 304.3, s = 'Sample Number = {0}\nRMSE ={1} K\nCorr. ={2}'.format(len(np.concatenate(rmse_list[3])[slice_nan]), format(rmse, '.4f'), format(corr, '.4f')), fontsize = 20)
            # text_sample_num.set_bbox(dict(facecolor = 'white', alpha = 1, edgecolor = 'white'))
            # text_rmse.set_bbox(dict(facecolor = 'white', alpha = 1, edgecolor = 'white'))
            text_pccs.set_bbox(dict(facecolor = 'white', alpha = 1, edgecolor = 'white'))
            colorbar = plt.colorbar()
            colorbar.ax.tick_params(labelsize = 20)
            plt.grid(color = 'grey', alpha = 0.3, linestyle = '--')
            plt.plot([292, 307], [292, 307], c = 'black', linestyle = '--', linewidth = 3)
            ax = plt.gca()
            plt.xticks(np.arange(292, 307.1, 3))
            plt.yticks(np.arange(292, 307.1, 3))
            ax.set_aspect(1)
            plt.xlim(292, 307)
            plt.ylim(292, 307)
            plt.clim(1, 100)
            plt.tick_params(labelsize = 20)
            plt.xlabel('TAO SST (K)', fontsize = 25)
            plt.ylabel('{0} SST (K)'.format(ylabel_list[plt_num]), fontsize = 25)
            plt.title('({0})'.format(order[order_list[plt_num]]), loc = 'left', fontsize = 35, weight = 'bold')
            plt.savefig(plotdir + '/TAO_{0}_SST'.format(ylabel_list[plt_num]), dpi = 1000, bbox_inches = 'tight')

    def plot_part(date, orbit, extent, ticks_interval, pcolor_vmin_ssws, pcolor_vmax_ssws, pcolor_vmin_sst, pcolor_vmax_sst, order_list):
        plotdir = plot_loc + '/Part/{0}_{1}'.format(date, orbit.capitalize())
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        # sws = xr.open_dataset(abspath + '/Note/Output/SWS0/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        # era5 = xr.open_dataset(abspath + '/Note/Output/ERA50/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        sws = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        era5 = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        sws_ssw = sws[:2]
        era_ssw = era5[:2]
        sws_sst = sws[2]
        era5_sst = era5[2]
        if orbit == 'asc':
            fy3_ssws = np.array(nc.Dataset(abspath + '/DataSet/Z/FY3D/SSW/FY3D_MWRIX_GBAL_L2_SWS_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date.replace('.', '')))['SWS_Ascending'])
        elif orbit == 'dsc':
            fy3_ssws = np.array(nc.Dataset(abspath + '/DataSet/Z/FY3D/SSW/FY3D_MWRIX_GBAL_L2_SWS_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date.replace('.', '')))['SWS_Descending'])
        fy3_ssws[np.where(fy3_ssws == -9999.)] = np.nan
        fy3_ssws[np.where(fy3_ssws == 110.0)] = np.nan
        fy3_ssws_temp = sws_sst.copy()
        fy3_ssws_temp[:] = fy3_ssws[(90 - 84) * 4:(90 + 84) * 4][::-1]
        fy3_ssws = fy3_ssws_temp.copy()
        fy3_ssws[:, :180 * 4] = np.array(fy3_ssws_temp[:, 180 * 4:])
        fy3_ssws[:, 180 * 4:] = np.array(fy3_ssws_temp[:, :180 * 4])
        if orbit == 'asc':
            fy3_sst = np.array(nc.Dataset(abspath + '/DataSet/Z/FY3D/SST/FY3D_MWRIX_GBAL_L2_SST_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date.replace('.', '')))['SST_Ascending']) / 100
        elif orbit == 'dsc':
            fy3_sst = np.array(nc.Dataset(abspath + '/DataSet/Z/FY3D/SST/FY3D_MWRIX_GBAL_L2_SST_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date.replace('.', '')))['SST_Descending']) / 100
        fy3_sst[np.where(fy3_sst == -99.99)] = np.nan
        fy3_sst_temp = sws_sst.copy()
        fy3_sst_temp[:] = fy3_sst[(90 - 84) * 4:(90 + 84) * 4][::-1]
        fy3_sst = fy3_sst_temp.copy()
        fy3_sst[:, :180 * 4] = np.array(fy3_sst_temp[:, 180 * 4:])
        fy3_sst[:, 180 * 4:] = np.array(fy3_sst_temp[:, :180 * 4])
        var_list = [sws_ssw, era_ssw, fy3_ssws, sws_sst, era5_sst, fy3_sst]
        # if orbit == 'asc':
        #     title_list = ['DGPRN, SSW Ascending {0}UTC'.format(date), 'ERA5, SSW {0}UTC'.format(date), 'FY3D, SSW Ascending {0}UTC'.format(date), \
        #                 'DGPRN, SST Ascending {0}UTC'.format(date), 'ERA5, SST {0}UTC'.format(date), 'FY3D, SST Ascending {0}UTC'.format(date)]
        # elif orbit == 'dsc':
        #     title_list = ['DGPRN, SSW Descending {0}UTC'.format(date), 'ERA5, SSW {0}UTC'.format(date), 'FY3D, SSW Descending {0}UTC'.format(date), \
        #                 'DGPRN, SST Descending {0}UTC'.format(date), 'ERA5, SST {0}UTC'.format(date), 'FY3D, SST Descending {0}UTC'.format(date)]
        title_list = ['DGPRN', 'ERA5', 'FY-3D']
        for var_num in range(len(var_list)):
            if order_list[var_num] != -1:
                var = var_list[var_num]
                if var_num in [0, 1]:
                    var[:, :, 180 * 4] = var[:, :, 180 * 4 + 1]
                    var[:, :, 180 * 4 - 1] = var[:, :, 180 * 4 + 1]
                    var[:, :, 180 * 4 - 2] = var[:, :, 179 * 4 - 1]
                    var[:, :, 180 * 4 - 3] = var[:, :, 179 * 4 - 1]
                    var[:, :, 179 * 4] = var[:, :, 179 * 4 - 1]
                    var = var.loc[:, extent[2]:extent[3], extent[0]:extent[1]]
                elif var_num in [3, 4]:
                    var[:, 180 * 4] = var[:, 180 * 4 + 1]
                    var[:, 180 * 4 - 1] = var[:, 180 * 4 + 1]
                    var[:, 180 * 4 - 2] = var[:, 179 * 4 - 1]
                    var[:, 180 * 4 - 3] = var[:, 179 * 4 - 1]
                    var[:, 179 * 4] = var[:, 179 * 4 - 1]
                    var = var.loc[extent[2]:extent[3], extent[0]:extent[1]]
                else:
                    var = var.loc[extent[2]:extent[3], extent[0]:extent[1]]
                lat = np.array(var['lat'])
                lon = np.array(var['lon'])
                fig = plt.figure(figsize = (8, 8))
                ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5)
                ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
                ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs = ccrs.PlateCarree())
                if extent[0] < 180:
                    ax.set_xticks(np.arange(extent[0], extent[1] + 0.1, ticks_interval))
                else:
                    ax.set_xticks(np.arange(extent[0] - 360, extent[1] - 360 + 0.1, ticks_interval))
                ax.set_yticks(np.arange(extent[2], extent[3] + 0.1, ticks_interval))
                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.tick_params(axis = 'both', which = 'major', labelsize = 20, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
                if var_num in [0, 1, 2]:
                    if var_num in [0, 1]:
                        pcolor = plt.pcolor(lon, lat, (np.array(var[0]) ** 2 + np.array(var[1]) ** 2) ** 0.5, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18, vmin = pcolor_vmin_ssws, vmax = pcolor_vmax_ssws)
                    else:
                        pcolor = plt.pcolor(lon, lat, np.array(var), transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18, vmin = pcolor_vmin_ssws, vmax = pcolor_vmax_ssws)
                    colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.07, extend = 'max')
                    colorbar.set_ticks(np.arange(pcolor_vmin_ssws, pcolor_vmax_ssws + 0.1, 3))
                    colorbar.set_label(label = 'Wind Speed (m/s)', fontsize = 25)
                    colorbar.ax.tick_params(labelsize = 20)
                elif var_num in [3, 4, 5]:
                    pcolor = plt.pcolor(lon, lat, np.array(var), transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18, vmin = pcolor_vmin_sst, vmax = pcolor_vmax_sst)
                    colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.07, extend = 'max')
                    colorbar.set_ticks(np.arange(pcolor_vmin_sst, pcolor_vmax_sst + 0.1, 1))
                    colorbar.set_label(label = 'Temperature (K)', fontsize = 25)
                    colorbar.ax.tick_params(labelsize = 20)
                quiver_interval = len(lon) // 24
                if var_num in [0, 1]:
                    plt.quiver(lon[::quiver_interval], lat[::quiver_interval], np.array(var[0])[::quiver_interval, ::quiver_interval], np.array(var[1])[::quiver_interval, ::quiver_interval], transform = ccrs.PlateCarree())
                plt.title(title_list[var_num], loc = 'center', fontsize = 25)
                plt.title('({0})'.format(order[order_list[var_num]]), loc = 'left', fontsize = 35, weight = 'bold')
                plt.savefig(plotdir + '/{0}_{1}-{2}_{3}.png'.format(order[order_list[var_num]], int(lat[0]), int(lon[0]), title_list[var_num]), dpi = 1000, bbox_inches = 'tight')

    def plot_part_res(date, orbit, extent, ticks_interval, pcolor_vmin_ssws, pcolor_vmax_ssws, pcolor_vmin_sst, pcolor_vmax_sst, order_list):
        plotdir = plot_loc + '/Res/{0}_{1}'.format(date, orbit.capitalize())
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        sws = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        sws_res = xr.open_dataset(abspath + '/Note/Output/SWS0/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        era5 = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        sws_ssw = sws[:2]
        era_ssw = era5[:2]
        sws_ssw_res = sws_res[:2]
        var_list = [sws_ssw, era_ssw, sws_ssw_res]
        title_list = ['DGPRN', 'ERA5', 'DGPRN without Resblock']
        for var_num in range(len(var_list)):
            if order_list[var_num] != -1:
                var = var_list[var_num]
                var = var.loc[:, extent[2]:extent[3], extent[0]:extent[1]]
                lat = np.array(var['lat'])
                lon = np.array(var['lon'])
                fig = plt.figure(figsize = (8, 8))
                ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5)
                ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
                ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs = ccrs.PlateCarree())
                if extent[0] < 180:
                    ax.set_xticks(np.arange(extent[0], extent[1] + 0.1, ticks_interval))
                else:
                    ax.set_xticks(np.arange(extent[0] - 360, extent[1] - 360 + 0.1, ticks_interval))
                ax.set_yticks(np.arange(extent[2], extent[3] + 0.1, ticks_interval))
                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.tick_params(axis = 'both', which = 'major', labelsize = 20, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
                if var_num in [0, 1, 2]:
                    pcolor = plt.pcolor(lon, lat, (np.array(var[0]) ** 2 + np.array(var[1]) ** 2) ** 0.5, transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18, vmin = pcolor_vmin_ssws, vmax = pcolor_vmax_ssws)
                    colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.07, extend = 'max')
                    colorbar.set_ticks(np.arange(pcolor_vmin_ssws, pcolor_vmax_ssws + 0.1, 3))
                    colorbar.set_label(label = 'Wind Speed (m/s)', fontsize = 25)
                    colorbar.ax.tick_params(labelsize = 20)
                elif var_num in [3, 4, 5]:
                    pcolor = plt.pcolor(lon, lat, np.array(var), transform = ccrs.PlateCarree(), cmap = cmaps.BlueDarkRed18, vmin = pcolor_vmin_sst, vmax = pcolor_vmax_sst)
                    colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.07, extend = 'max')
                    colorbar.set_ticks(np.arange(pcolor_vmin_sst, pcolor_vmax_sst + 0.1, 1))
                    colorbar.set_label(label = 'Temperature (K)', fontsize = 25)
                    colorbar.ax.tick_params(labelsize = 20)
                quiver_interval = len(lon) // 24
                plt.quiver(lon[::quiver_interval], lat[::quiver_interval], np.array(var[0])[::quiver_interval, ::quiver_interval], np.array(var[1])[::quiver_interval, ::quiver_interval], transform = ccrs.PlateCarree())
                plt.title(title_list[var_num], loc = 'center', fontsize = 25)
                plt.title('({0})'.format(order[order_list[var_num]]), loc = 'left', fontsize = 35, weight = 'bold')
                plt.savefig(plotdir + '/{0}_{1}-{2}_{3}.png'.format(order[order_list[var_num]], int(lat[0]), int(lon[0]), title_list[var_num]), dpi = 1000, bbox_inches = 'tight')

    def plot_concat(date, orbit, order_list):
        plotdir = plot_loc + '/Concat/{0}_{1}'.format(date, orbit.capitalize())
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        sws = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        era5 = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        sws_ssw = sws[:2]
        era_ssw = era5[:2]
        sws_sst = sws[2]
        era5_sst = era5[2]

        slice_nan = np.where(np.isnan(np.array(sws_ssw)) == True)

        if orbit == 'asc':
            fy3_ssws = np.array(nc.Dataset(abspath + '/DataSet/Z/FY3D/SSW/FY3D_MWRIX_GBAL_L2_SWS_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date.replace('.', '')))['SWS_Ascending'])
        elif orbit == 'dsc':
            fy3_ssws = np.array(nc.Dataset(abspath + '/DataSet/Z/FY3D/SSW/FY3D_MWRIX_GBAL_L2_SWS_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date.replace('.', '')))['SWS_Descending'])
        fy3_ssws[np.where(fy3_ssws == -9999.)] = np.nan
        fy3_ssws[np.where(fy3_ssws == 110.0)] = np.nan
        fy3_ssws_temp = sws_sst.copy()
        fy3_ssws_temp[:] = fy3_ssws[(90 - 84) * 4:(90 + 84) * 4][::-1]
        fy3_ssws = fy3_ssws_temp.copy()
        fy3_ssws[:, :180 * 4] = np.array(fy3_ssws_temp[:, 180 * 4:])
        fy3_ssws[:, 180 * 4:] = np.array(fy3_ssws_temp[:, :180 * 4])
        if orbit == 'asc':
            fy3_sst = np.array(nc.Dataset(abspath + '/DataSet/Z/FY3D/SST/FY3D_MWRIX_GBAL_L2_SST_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date.replace('.', '')))['SST_Ascending']) / 100
        elif orbit == 'dsc':
            fy3_sst = np.array(nc.Dataset(abspath + '/DataSet/Z/FY3D/SST/FY3D_MWRIX_GBAL_L2_SST_MLT_GLL_{0}_POAD_025KM_MS.HDF'.format(date.replace('.', '')))['SST_Descending']) / 100
        fy3_sst[np.where(fy3_sst == -99.99)] = np.nan
        fy3_sst_temp = sws_sst.copy()
        fy3_sst_temp[:] = fy3_sst[(90 - 84) * 4:(90 + 84) * 4][::-1]
        fy3_sst = fy3_sst_temp.copy()
        fy3_sst[:, :180 * 4] = np.array(fy3_sst_temp[:, 180 * 4:])
        fy3_sst[:, 180 * 4:] = np.array(fy3_sst_temp[:, :180 * 4])
        var_list = [sws_ssw, era_ssw, fy3_ssws, sws_sst, era5_sst, fy3_sst]
        if orbit == 'asc':
            title_list = ['DGPRN, {0}UTC'.format(date), 'ERA5, {0}UTC'.format(date), 'FY-3D, {0}UTC'.format(date), \
                        'DGPRN, SST Ascending {0}UTC'.format(date), 'ERA5, SST {0}UTC'.format(date), 'FY-3D, SST Ascending {0}UTC'.format(date)]
        elif orbit == 'dsc':
            title_list = ['DGPRN, {0}UTC'.format(date), 'ERA5, {0}UTC'.format(date), 'FY-3D, {0}UTC'.format(date), \
                        'DGPRN, SST Descending {0}UTC'.format(date), 'ERA5, SST {0}UTC'.format(date), 'FY-3D, SST Descending {0}UTC'.format(date)]
        for var_num in range(len(var_list)):
            if order_list[var_num] != -1:
                var = var_list[var_num]
                lat = np.array(var['lat'])
                lon = np.array(var['lon'])
                if var_num in [0, 1]:
                    var = np.array(var)
                    var[slice_nan] = np.nan
                    var = ((var[0] ** 2) + (var[1] ** 2)) ** 0.5
                if orbit == 'asc':
                    if var_num in [0, 1, 3, 4]:
                        var[:, 180 * 4] = var[:, 180 * 4 + 1]
                        var[:, 180 * 4 - 1] = var[:, 180 * 4 + 1]
                        var[:, 180 * 4 - 2] = var[:, 179 * 4 - 1]
                        var[:, 180 * 4 - 3] = var[:, 179 * 4 - 1]
                        var[:, 179 * 4] = var[:, 179 * 4 - 1]
                        var_temp = var.copy()
                        window = np.ones(int(5)) / float(5)
                        for lat_num in range(len(lat)):
                            var_temp[lat_num] = np.convolve(var_temp[lat_num], window, 'same')
                        for lon_convolve in [180]:
                            part = var[:, int((lon_convolve - 1) * 4 + 1):int((lon_convolve + 1) * 4 + 1)]
                            part_convolve = var_temp[:, int((lon_convolve - 1) * 4 + 1):int((lon_convolve + 1) * 4 + 1)]
                            for lat_num in range(part.shape[0]):
                                for lon_num in range(part.shape[1]):
                                    part[lat_num, lon_num] = part_convolve[lat_num, lon_num]
                            var[:, int((lon_convolve - 1) * 4 + 1):int((lon_convolve + 1) * 4 + 1)] = part
                elif orbit == 'dsc':
                    if var_num in [0, 1, 3, 4]:
                        var[:, 180 * 4] = var[:, 180 * 4 - 1]
                        var[:, 180 * 4 + 1] = var[:, 180 * 4 - 1]
                        var[:, 180 * 4 + 2] = var[:, 181 * 4 + 1]
                        var[:, 180 * 4 + 3] = var[:, 181 * 4 + 1]
                        var[:, 181 * 4] = var[:, 181 * 4 + 1]
                        var_temp = var.copy()
                        window = np.ones(int(5)) / float(5)
                        for lat_num in range(len(lat)):
                            var_temp[lat_num] = np.convolve(var_temp[lat_num], window, 'same')
                        for lon_convolve in [181]:
                            part = var[:, int((lon_convolve - 1) * 4 + 1):int((lon_convolve + 1) * 4 + 1)]
                            part_convolve = var_temp[:, int((lon_convolve - 1) * 4 + 1):int((lon_convolve + 1) * 4 + 1)]
                            for lat_num in range(part.shape[0]):
                                for lon_num in range(part.shape[1]):
                                    part[lat_num, lon_num] = part_convolve[lat_num, lon_num]
                            var[:, int((lon_convolve - 1) * 4 + 1):int((lon_convolve + 1) * 4 + 1)] = part
                        var_temp = var.copy()
                        window = np.ones(int(5)) / float(5)
                        for lon_num in range(len(lon)):
                            var_temp[:, lon_num] = np.convolve(var[:, lon_num], window, 'same')
                        for lat_convolve in [85]:
                            part = var[int((lat_convolve - 2) * 4):int((lat_convolve + 2) * 4)]
                            part_convolve = var_temp[int((lat_convolve - 2) * 4):int((lat_convolve + 2) * 4)]
                            for lat_num in range(part.shape[0]):
                                for lon_num in range(part.shape[1]):
                                    if np.isnan(part_convolve[lat_num, lon_num]) == False:
                                        part[lat_num, lon_num] = part_convolve[lat_num, lon_num]
                            var[int((lat_convolve - 2) * 4):int((lat_convolve + 2) * 4)] = part
                var, lon = add_cyclic_point(var, coord = lon)
                fig = plt.figure(figsize = (8, 8))
                ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree(central_longitude = 180))
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5)
                ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
                ax.set_extent([-180, 180, -90, 90], crs = ccrs.PlateCarree(central_longitude = 180))
                ax.set_xticks(range(-180, 180 + 1, 40))
                ax.set_yticks(range(-90, 90 + 1, 20))
                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.tick_params(axis = 'both', which = 'major', labelsize = 15, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
                if var_num in [0, 1, 2]:
                    pcolor = plt.pcolor(lon - 180, lat, np.array(var), transform = ccrs.PlateCarree(central_longitude = 180), cmap = cmaps.BlueDarkRed18, vmin = 0, vmax = 18)
                    colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05, extend = 'max')
                    colorbar.set_ticks(np.arange(0, 18.1, 3))
                    colorbar.set_label(label = 'Wind Speed (m/s)', fontsize = 25)
                    colorbar.ax.tick_params(labelsize = 20)
                elif var_num in [3, 4, 5]:
                    pcolor = plt.pcolor(lon - 180, lat, np.array(var), transform = ccrs.PlateCarree(central_longitude = 180), cmap = cmaps.BlueDarkRed18, vmin = 272, vmax = 302)
                    colorbar = fig.colorbar(pcolor, shrink = 0.7, orientation = 'horizontal', pad = 0.05, extend = 'max')
                    colorbar.set_ticks(np.arange(272, 302.1, 5))
                    colorbar.set_label(label = 'Temperature (K)', fontsize = 25)
                    colorbar.ax.tick_params(labelsize = 20)
                plt.title(title_list[var_num], loc = 'center', fontsize = 25)
                plt.title('({0})'.format(order[order_list[var_num]]), loc = 'left', fontsize = 35, weight = 'bold')
                plt.savefig(plotdir + '/{0}_{1}.png'.format(order[order_list[var_num]], title_list[var_num]), dpi = 1000, bbox_inches = 'tight')

    def plot_res(date, orbit, extent, ticks_interval, order_list):
        plotdir = plot_loc + '/Res/{0}_{1}'.format(date, orbit.capitalize())
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        color_list = ['000,000,000', '244,111,068', '075,101,175']
        zorder_list = [30, 20, 10]
        sws = xr.open_dataset(abspath + '/Note/Output/SWS/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        era5 = xr.open_dataset(abspath + '/Note/Output/ERA5/ERA5_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        sws_res = xr.open_dataset(abspath + '/Note/Output/SWS0/SWS_{0}_sswsst_{1}.nc'.format(orbit, date))['sswsst'][0]
        sws_ssw = sws[:2]
        era5_ssw = era5[:2]
        sws_ssw_res = sws_res[:2]
        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5)
        ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs = ccrs.PlateCarree())
        if extent[0] < 180:
            ax.set_xticks(np.arange(extent[0], extent[1] + 0.1, ticks_interval))
        else:
            ax.set_xticks(np.arange(extent[0] - 360, extent[1] - 360 + 0.1, ticks_interval))
        ax.set_yticks(np.arange(extent[2], extent[3] + 0.1, ticks_interval))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
        var_list = [era5_ssw, sws_ssw, sws_ssw_res]
        if orbit == 'asc':
            title_list = ['DGPRN, SSW Ascending', 'ERA5, SSW', 'DGPRN, SSW Ascending', 'DGPRN, SST Ascending', 'ERA5, SST', 'DGPRN, SST Ascending']
        elif orbit == 'dsc':
            title_list = ['DGPRN, SSW Descending', 'ERA5, SSW', 'DGPRN, SSW Descending', 'DGPRN, SST Descending', 'ERA5, SST', 'DGPRN, SST Descending']
        label_list = ['ERA5', 'DGPRN', 'DGPRN without Resblock']
        for var_num in range(len(var_list)):
            var = var_list[var_num]
            var = var.loc[:, extent[2]:extent[3], extent[0]:extent[1]]
            lat = np.array(var['lat'])
            lon = np.array(var['lon'])
            quiver_interval = len(lon) // 16
            plt.quiver(lon[::quiver_interval], lat[::quiver_interval], np.array(var[0])[::quiver_interval, ::quiver_interval], np.array(var[1])[::quiver_interval, ::quiver_interval], transform = ccrs.PlateCarree(), \
                       color = rgb_to_hex(color_list[var_num]), label = label_list[var_num], zorder = zorder_list[var_num])
        # plt.title(title_list[var_num], loc = 'center', fontsize = 25)
        plt.title('({0})'.format(order[order_list[0]]), loc = 'left', fontsize = 35, weight = 'bold')
        plt.savefig(plotdir + '/Res_{0}_{1}-{2}_{3}.png'.format(order[order_list[0]], int(lat[0]), int(lon[0]), title_list[var_num]), dpi = 1000, bbox_inches = 'tight')
    
        plt.figure(figsize = (8, 8))
        ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5)
        ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'lightgrey')
        ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs = ccrs.PlateCarree())
        if extent[0] < 180:
            ax.set_xticks(np.arange(extent[0], extent[1] + 0.1, ticks_interval))
        else:
            ax.set_xticks(np.arange(extent[0] - 360, extent[1] - 360 + 0.1, ticks_interval))
        ax.set_yticks(np.arange(extent[2], extent[3] + 0.1, ticks_interval))
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
        var_list = [era5_ssw, sws_ssw, sws_ssw_res]
        if orbit == 'asc':
            title_list = ['DGPRN, SSW Ascending', 'ERA5, SSW', 'DGPRN, SSW Ascending', 'DGPRN, SST Ascending', 'ERA5, SST', 'DGPRN, SST Ascending']
        elif orbit == 'dsc':
            title_list = ['DGPRN, SSW Descending', 'ERA5, SSW', 'DGPRN, SSW Descending', 'DGPRN, SST Descending', 'ERA5, SST', 'DGPRN, SST Descending']
        label_list = ['ERA5', 'DGPRN', 'DGPRN without Resblock']
        for var_num in range(len(var_list)):
            plt.quiver(np.nan, np.nan, np.nan, np.nan, transform = ccrs.PlateCarree(), color = rgb_to_hex(color_list[var_num]), label = label_list[var_num])
        # plt.title(title_list[var_num], loc = 'center', fontsize = 25)
        plt.legend(frameon = True, framealpha = 1, ncol = 3, fontsize = 25)
        plt.title('({0})'.format(order[order_list[0]]), loc = 'left', fontsize = 35, weight = 'bold')
        plt.savefig(plotdir + '/Legend.png', dpi = 1000, bbox_inches = 'tight')

    plot_loc = abspath + '/Save/Plot/Conclusion'
    domain_list = [[-84, -45], [-45, -30], [-30, -15], [-15, 0], [0, 15], [15, 30], [30, 45], [45, 84], [-84, 84]]
    begin_date = '2022-01-01'
    end_date = '2023-12-01'
    date_list = []
    begin_datetime = datetime.datetime.strptime(begin_date, '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    while begin_datetime <= end_datetime:
        date_str = begin_datetime.strftime('%Y.%m.%d')
        if date_str not in  ['2022.01.07', '2022.12.14', '2023.01.07']:
            date_list.append(date_str)
        begin_datetime = begin_datetime + datetime.timedelta(days = 1)

    # plot_error_rmse()

    # plot_error_corr()

    # plot_rmse_spatial()

    plot_rmse_tao_ssw([1, 0, 2])

    # plot_rmse_tao_sst([4, 3, 5])

    # plot_part('2023.05.24', [140, 160, 32, 52], 0, 32, 270, 280)
    # plot_part('2023.07.27', [110, 130, 7, 27], 0, 24, 300, 303)
    # plot_part('2023.08.12', [110, 160, 0, 40], 0, 24, 300, 303)

    # plot_part('2023.07.31', 'asc', [124, 144, 12, 32], 4, 0, 27, 300, 303, [1, 0, 2, -1, -1, -1])
    # plot_part('2023.07.31', 'asc', [106, 126, -20, 0], 4, 0, 10, 300, 303, [1, 0, 2, -1, -1, -1])
    # plot_part('2023.10.02', 'dsc', [116, 136, 10, 30], 4, 0, 27, 300, 303, [4, 3, 5, -1, -1, -1])
    # plot_part('2023.10.02', 'dsc', [320, 360, 20, 60], 8, 0, 18, 300, 303, [4, 3, 5, -1, -1, -1])

    # plot_concat('2023.07.31', 'asc', [1, 0, -1, -1, -1, -1])
    # plot_concat('2023.10.02', 'dsc', [3, 2, -1, -1, -1, -1])

    # plot_part_res('2023.07.31', 'asc', [110, 122, -12, 0], 2, 0, 9, 300, 303, [1, 0, 2, -1, -1, -1])
    # plot_part_res('2023.10.02', 'dsc', [330, 350, 40, 60], 4, 0, 18, 300, 303, [4, 3, 5, -1, -1, -1])

    # plot_res('2023.07.31', 'asc', [110, 122, -12, 0], 2, [0])
    # plot_res('2023.10.02', 'dsc', [330, 350, 40, 60], 4, [1])