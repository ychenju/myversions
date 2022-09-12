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

def datafilter(filtee, sifter, func):
    r = copy.deepcopy(filtee)
    for i,x in enumerate(filtee):
        for j,y in enumerate(x):
            r[i,j] = func(y, sifter[i,j])
    return np.array(r)

class filterf:
    @staticmethod
    def nansync(x, f):
        if np.isnan(f):
            return np.nan
        else:
            return x

    @staticmethod
    def falsetonan(x, f):
        if f:
            return x
        else:
            return np.nan
    

class wrfout:
    ncfile = None

    def __init__(self, path):
        self.ncfile = xr.open_dataset(path)
    
    def extract(self, *keys):
        return frame(self.ncfile, *keys)   # NOT: return frame(self, keys) ==> *keys is several variables, keys is a tuple

    def __getitem__(self, key):
        return np.array(self.ncfile[key]).squeeze()
    
FRAME_DEFAULT_FLAGS = {
    'REMOVEWATER': False,
}

class frame:
    lat = None
    long = None
    time = None

    def __init__(self, source, *keys):
        self.__data = {}    #   length-changeable variables
        self.__flag = FRAME_DEFAULT_FLAGS    #   store flags
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

    def getflag(self, key):
        return self.__flag[key]

    def setflag(self, key, value):
        self.__flag[key] = value
        return self.__flag[key]

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

    '''
    # old version
    def removewater(self):
        if not ('LANDMASK' in self.__data.keys()):
            raise RuntimeError("'LANDMASK' has not been loaded to the data frame")
        mask = self['LANDMASK']
        ori = self.lat
        for k in self.getall().keys():
            if self[k].shape == ori.shape:
                self[k] = _lmf(self.get(k), mask)
    '''

    def removewater(self):
        if not ('LANDMASK' in self.__data.keys()):
            raise RuntimeError("'LANDMASK' has not been loaded to the data frame")
        ori = self.lat
        for k in filter(lambda _x: _x != 'LANDMASK', self.getall().keys()):
            if self[k].shape == ori.shape:
                self[k] = datafilter(self[k], self['LANDMASK'], filterf.falsetonan)
        self['XLAT_RW'] = datafilter(self.lat, self['LANDMASK'], filterf.falsetonan)
        self['XLONG_RW'] = datafilter(self.long, self['LANDMASK'], filterf.falsetonan)
        self.__flag['REMOVEWATER'] = True

    def removewater_sub(self):
        if not ('LANDMASK' in self.__data.keys()):
            raise RuntimeError("'LANDMASK' has not been loaded to the data frame")
        ori = self.lat
        for k in self.getall().keys().remove('LANDMASK'):
            if self[k].shape == ori.shape:
                self[k] = datafilter(self[k], sifter=self['LANDMASK'], func=filterf.falsetonan)

    def planefit(self, key):
        if self.__flag['REMOVEWATER']:
            _T1, _Tr = sc.fit(self.__data['XLONG_RW'], self.__data['XLAT_RW'], self.__data[key])
        else:
            _T1, _Tr = sc.fit(self.long, self.lat, self.__data[key])
        self.__data['PF_'+key+'_1'] = _T1
        self.__data['PF_'+key+'_r'] = _Tr
        return _T1, _Tr

    def quickshow(self, key):
        _m = bmap()
        _m.quickcountourf(self, key)
        plt.title(key)
        plt.colorbar()
        plt.show()

BMAP_DEFAULT_ATTRS = {
    'figsize'   :   (12,8),
    'dpi'       :   180,
    'proj'      :   'cyl',          # changed from 'cea'
    'lon'       :   [-180.,180.],
    'lat'       :   [-60.,60.],
    'res'       :   'c',
    'latinv'    :   0.,
    'loninv'    :   0.,
    'fontsize'  :   10,
    'cmap'      :   'jet',          # default colormap
    'clcolor'   :   'k',            # coastline $color
    'cllw'      :   1.,             # coastline $linewidth
    'clbgs'     :   0.2,            # colored backgrounds $scale
    'gridc'     :   'k',       # color of parallels and meridians
    'gridlw'    :   0.5,            # linewidth of parallels and meridians
}    

class bmap:
    # __attr = {  'figsize'   :   (12,8),
    #             'dpi'       :   180,
    #             'proj'      :   'cea',
    #             'lon'       :   [-180.,180.],
    #             'lat'       :   [-60.,60.],
    #             'res'       :   'c',
    #             'latinv'    :   0.,
    #             'loninv'    :   0.,
    #             'fontsize'  :   10,
    #             'cmap'      :   'jet',
    #             }

    # __base = None

    def __init__(self, **kw):
        self.__attr = BMAP_DEFAULT_ATTRS
        # self.__attr = {
        #     'figsize'   :   (12,8),
        #         'dpi'       :   180,
        #         'proj'      :   'cea',
        #         'lon'       :   [-180.,180.],
        #         'lat'       :   [-60.,60.],
        #         'res'       :   'c',
        #         'latinv'    :   0.,
        #         'loninv'    :   0.,
        #         'fontsize'  :   10,
        #         'cmap'      :   'jet',
        # }
        self.__base = None
        for k in kw:
            self.__attr[k] = kw[k]

    def resetall(self):
        self.__attr = BMAP_DEFAULT_ATTRS
        self.__base = None

    def reset(self, **kw):
        self.resetall()
        for k in kw:
            self.__attr[k] = kw[k]
    
    def set(self, **kw):
        for k in kw:
            self.__attr[k] = kw[k]
    

    def bg(self):
        plt.figure(figsize=self.__attr['figsize'], dpi=self.__attr['dpi'])
        self.__base = Basemap(projection=self.__attr['proj'], llcrnrlon = self.__attr['lon'][0], llcrnrlat = self.__attr['lat'][0], 
                    urcrnrlon = self.__attr['lon'][1], urcrnrlat = self.__attr['lat'][1], resolution=self.__attr['res'])
        self.__base.drawcoastlines(color=self.__attr['clcolor'], linewidth=self.__attr['cllw'])

        if int(self.__attr['latinv']):
            self.__base.drawparallels(np.arange(self.__attr['lat'][0], self.__attr['lat'][1], self.__attr['latinv']), labels=[1,0,0,0],
                                        color=self.__attr['gridc'], linewidth=self.__attr['gridlw'], fontsize=self.__attr['fontsize'])
        if int(self.__attr['loninv']):
            self.__base.drawmeridians(np.arange(self.__attr['lon'][0], self.__attr['lon'][1], self.__attr['loninv']), labels=[0,0,0,1],
                                        color=self.__attr['gridc'], linewidth=self.__attr['gridlw'], fontsize=self.__attr['fontsize'])

    def colorbg(self, style=None):
        if style == 'bluemarble':
            self.__base.bluemarble(scale=self.__attr['clbgs'])
        if style == 'shadedrelief':
            self.__base.shadedrelief(scale=self.__attr['clbgs'])
        if style == 'etopo':
            self.__base.etopo(scale=self.__attr['clbgs'])

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
