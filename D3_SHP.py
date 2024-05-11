import os
import numpy as np
import xarray as xr

abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
topography = xr.open_dataset('F:/SHP/TOPOGRAPHY/ETOPO2v2cF4_Topography.nc')['z']
topography = topography.rename({'y':'lat', 'x':'lon'})
topography = topography.interp(lat = np.arange(-90, 90.001, 0.125), lon = np.arange(-180, 180.001, 0.125), method = 'linear')
topography = topography.interpolate_na(dim = 'lat', method = 'linear', fill_value = 'extrapolate')
topography = topography.interpolate_na(dim = 'lon', method = 'linear', fill_value = 'extrapolate')
topography = topography[::-1]
topography = (topography - np.min(topography)) / (np.max(topography) - np.min(topography))
topography = topography.astype(dtype = np.float32)
print(topography)
print(topography.shape)
topography = xr.Dataset({'topography':topography})
topography.to_netcdf(abspath + '/DataSet/X/SHP/SHP_topography.nc')