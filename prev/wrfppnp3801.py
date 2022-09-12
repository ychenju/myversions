# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sigmacalc as sc
import copy

def _lm_filter(x, lm):
    if lm:
        return x
    else:
        return np.nan

def _lmf(ar, lmar):
    r = []
    for i,x in enumerate(ar):
        r.append([])
        for j,y in enumerate(x):
            r[-1].append(_lm_filter(y, lmar[i][j]))
    return np.array(r)

class wrfout:
    ncfile = None

    def __init__(self, path):
        self.ncfile = xr.open_dataset(path)
    
    def extract(self, *keys):
        return frame(self.ncfile, *keys)   # NOT: return frame(self, keys) ==> *keys is several variables, keys is a tuple

    def __getitem__(self, key):
        return np.array(self.ncfile[key]).squeeze()
    

class frame:
    lat = None
    long = None
    time = None

    def __init__(self, source, *keys):
        self.__data = {}    #   length-changeable variables
        self.lat =  copy.deepcopy(np.array(source['XLAT']).squeeze())
        self.long = copy.deepcopy(np.array(source['XLONG']).squeeze())
        self.time = copy.deepcopy(np.array(source['XTIME']).squeeze())
        for key in keys:
            self.__data[key] = copy.deepcopy(np.array(source[key]).squeeze())
    
    def load(self, source, *keys):    # source is a wrfout
        for key in keys:
            self.__data[key] = copy.deepcopy(np.array(source.ncfile[key]).squeeze())

    def params(self):   # print the parameters loaded
        _l = [x for x in self.__data]
        print(_l)
        return _l
    
    def get(self, key): # return any parameter loaded
        return self.__data[key]
    
    def __getitem__(self, key):
        return self.__data[key]
    
    def getall(self):  # return __data
        return self.__data

    def set(self, key, value):  # change the parameter
        self.__data[key] = value
        return self.__data[key]
    
    def __setitem__(self, key, value):
        self.__data[key] = value
        return self.__data[key]
    
    def delete(self, key):  # delete any parameter
        del self.__data[key]
    
    def __delitem__(self, key):
        del self.__data[key]

    '''
    # operations
    def removewater_new(self):
        if not ('LANDMASK' in self.__data.keys()):
            self.load('LANDMASK')
        r = wrfout(source=False, parent=self)
        mask = self.get('LANDMASK')
        for k in r.getall().keys():
            if r.get(k).shape == self.lat.shape or r.get(k).shape == self.long.shape:
                r.set(k, _lmf(r.get(k), mask))
        return r
    '''

    def removewater(self):
        if not ('LANDMASK' in self.__data.keys()):
            raise RuntimeError("'LANDMASK' has not been loaded to the data frame")
        mask = self['LANDMASK']
        ori = self.lat
        for k in self.getall().keys():
            if self[k].shape == ori.shape:
                self[k] = _lmf(self.get(k), mask)

        

class bmap:
    __attr = {  'figsize'   :   (12,8),
                'dpi'       :   180,
                'proj'      :   'cea',
                'lon'       :   [-180.,180.],
                'lat'       :   [-60.,60.],
                'res'       :   'c',
                'latinv'    :   0.,
                'loninv'    :   0.,
                'fontsize'  :   10,
                'cmap'      :   'jet',
                }

    __base = None

    def __init__(self, **kw):
        for k in kw:
            self.__attr[k] = kw[k]
    
    def set(self, **kw):
        for k in kw:
            self.__attr[k] = kw[k]

    def bg(self):
        plt.figure(figsize=self.__attr['figsize'], dpi=self.__attr['dpi'])
        self.__base = Basemap(projection=self.__attr['proj'], llcrnrlon = self.__attr['lon'][0], llcrnrlat = self.__attr['lat'][0], 
                    urcrnrlon = self.__attr['lon'][1], urcrnrlat = self.__attr['lat'][1], resolution=self.__attr['res'])
        self.__base.drawcoastlines()

        if int(self.__attr['latinv']):
            self.__base.drawparallels(np.arange(self.__attr['lat'][0], self.__attr['lat'][1], self.__attr['latinv']), labels=[1,0,0,0], fontsize=self.__attr['fontsize'])
        if int(self.__attr['loninv']):
            self.__base.drawmeridians(np.arange(self.__attr['lon'][0], self.__attr['lon'][1], self.__attr['loninv']), labels=[0,0,0,1], fontsize=self.__attr['fontsize'])

    def fitboundaries(self, source):
        self.__attr['lon'] = [source.long[0,:].min(), source.long[0,:].max()]
        self.__attr['lat'] = [source.lat[:,0].min(), source.lat[:,0].max()]
    
    def womesh(self, source):
        _lon, _lat = np.meshgrid(source.long[0,:], source.lat[:,0])
        return self.__base(_lon, _lat)

    def quickcountourf(self, source, key):
        self.fitboundaries(source)
        self.bg()
        _X, _Y = self.womesh(source)
        if 'levels' in self.__attr.keys():
            self.__base.contourf(_X,_Y,source.get(key),cmap=self.__attr['cmap'],levels=self.__attr['levels'])
        else:
            self.__base.contourf(_X,_Y,source.get(key),cmap=self.__attr['cmap'])

