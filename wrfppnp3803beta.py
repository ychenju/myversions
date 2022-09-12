# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sigmacalc as sc
import copy
import emap101 as emap

def _lm_filter(x, lm):
    if lm:
        return x
    else:
        return np.nan

# def _lmf(ar, lmar):
#     r = []
#     for i,x in enumerate(ar):
#         r.append([])
#         for j,y in enumerate(x):
#             r[-1].append(_lm_filter(y, lmar[i][j]))
#     return np.array(r)

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

    @staticmethod
    def isnan(f):
        if np.isnan(f):
            return 1
        else:
            return 0
    

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
    'RES': 1.,
}

class frame:
    lat = None
    long = None
    time = None

    def __init__(self, source, *keys):
        self._data = {}    #   length-changeable variables
        #self._flag = copy.deepcopy(FRAME_DEFAULT_FLAGS)    #   store flags
        self._flag = {}
        for flag in FRAME_DEFAULT_FLAGS.keys():
            self._flag[flag] = FRAME_DEFAULT_FLAGS[flag]
        self._chara = {}   #   store characteristic parameters
        self.lat =  copy.deepcopy(np.array(source['XLAT']).squeeze())
        self.long = copy.deepcopy(np.array(source['XLONG']).squeeze())
        self.time = copy.deepcopy(np.array(source['XTIME']).squeeze())
        for key in keys:
            self._data[key] = copy.deepcopy(np.array(source[key]).squeeze())
    
    def load(self, source, *keys):    # source is a wrfout
        for key in keys:
            self._data[key] = copy.deepcopy(np.array(source.ncfile[key]).squeeze())

    def params(self):   # print the parameters loaded
        _l = [x for x in self._data]
        print(_l)
        return _l
    
    def get(self, key): # return any parameter loaded
        return self._data[key]
    
    def __getitem__(self, key):
        return self._data[key]
    
    def getall(self):  # return _data
        return self._data

    def set(self, key, value):  # change the parameter
        self._data[key] = value
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
        return self._data[key]
    
    def delete(self, key):  # delete any parameter
        del self._data[key]
    
    def __delitem__(self, key):
        del self._data[key]

    def getflag(self, key):
        return self._flag[key]

    def setflag(self, key, value):
        self._flag[key] = value
        return self._flag[key]

    def getchara(self, key):
        return self._chara[key]

    def cp2dattr(self, attr):
        r = []
        for i in range(attr.shape[0]):
            r.append([])
            for j in range(attr.shape[1]):
                r[-1].append(attr[i,j])
        return np.array(r)

    #   'chara' is obtained through calculations, therefore there is no 'setchara'

    def removewater(self):
        if not ('LANDMASK' in self._data.keys()):
            raise RuntimeError("'LANDMASK' has not been loaded to the data frame")
        ori = self.lat
        for k in filter(lambda _x: _x != 'LANDMASK', self.getall().keys()):
            if self[k].shape == ori.shape:
                self[k] = datafilter(self[k], self['LANDMASK'], filterf.falsetonan)
        self['XLAT_RW'] = datafilter(self.lat, self['LANDMASK'], filterf.falsetonan)
        self['XLONG_RW'] = datafilter(self.long, self['LANDMASK'], filterf.falsetonan)
        self._flag['REMOVEWATER'] = True

    def planefit(self, key):
        if self._flag['REMOVEWATER']:
            _T1, _Tr = sc.fit(self._data['XLONG_RW'], self._data['XLAT_RW'], self._data[key])
        else:
            _T1, _Tr = sc.fit(self.long, self.lat, self._data[key])
        self._data['PF_'+key+'_1'] = _T1
        self._data['PF_'+key+'_r'] = _Tr
        return _T1, _Tr

    def sigma(self, key):
        if self._flag['REMOVEWATER']:
            _sig = sc.sigma(self._data['XLONG_RW'], self._data['XLAT_RW'], self._data[key])
        else:
            _sig = sc.sigma(self.long, self.lat, self._data[key])
        self._chara[key+'_SIGMA'] = _sig
        return _sig

    def quickshow(self, key):
        _m = emap.emap()
        _m.quickcountourf(self, key)
        plt.title(key)
        plt.colorbar()
        plt.show()

    '''
    
    '''
    # get a 1d (len=9) array of the data in self[key] around (x,y) coordinations (3x3)
    def get3x3(self, key, x, y):
        return np.array([self[key][x+i,y+j] for i in (-1,0,1) for j in (-1,0,1)])

    def mean3x3(self):
        #   1. Generate a new data frame
        #   2. Copy the lat, long & time to the new frame
        r = voidFrame(self.lat, self.long, self.time)
        #   3. For each _data, calc 'mean 3x3'
        for key in self.getall().keys():
        #       3.1 anchor the crop area: [1:-1, 1:-1]
            d = []
            for i in range(self[key].shape[0]):
                d.append([])
                for j in range(self[key].shape[1]):
                    d[-1].append(self[key][i,j])

            # d = copy.deepcopy(self[key])
            for i, x in enumerate(d[1:-1]):
                for j, _ in enumerate(x[1:-1]):
        #       3.2 keep the value out of the crop area
        #       3.3 calc the 3x3 area mean
                    if np.mean(list(map(filterf.isnan, self.get3x3(key,i,j)))) < 0.5:
                        d[i+1][j+1] = np.nanmean(self.get3x3(key,i,j))
                    else:
                        d[i+1][j+1] = np.nan# np.mean(self.get3x3(key,i,j))
        #           3.3.1   How to deal with NAN ?
            r[key] = np.array(d)
        for flag in self._flag.keys():
            r._flag[flag] = self._flag[flag]
        return r

    '''
    这个函数
    '''
    def crop(self, interv=3, fromx=1, fromy=1, tox=-1, toy=-1):
        _r = voidFrame(self.lat[fromx:tox:interv, fromy:toy:interv], self.long[fromx:tox:interv, fromy:toy:interv], self.time)
        for key in self.getall().keys():
            _r[key] = self[key][fromx:tox:interv, fromy:toy:interv]
        for flag in self._flag.keys():
            _r._flag[flag] = self._flag[flag]
        _r._flag['RES'] *= interv
        return _r

    def lowres3(self, fromx=1, fromy=1, tox=-1, toy=-1):
        r = self.mean3x3()
        r = r.crop(3)
        return r

    def res(self):
        return self._flag['RES']

class voidFrame(frame):
    def __init__(self, lat, long, time):
        self.lat = self.cp2dattr(lat)
        self.long = self.cp2dattr(long)
        self.time = time
        self._data = {}    #   length-changeable variables
        #self._flag = copy.deepcopy(FRAME_DEFAULT_FLAGS)    #   store flags
        self._flag = {}
        for flag in FRAME_DEFAULT_FLAGS.keys():
            self._flag[flag] = FRAME_DEFAULT_FLAGS[flag]
        self._chara = {}   #   store characteristic parameters

'''

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

    def __init__(self, **kw):
        self._attr = BMAP_DEFAULT_ATTRS
        self._base = None
        for k in kw:
            self._attr[k] = kw[k]

    def resetall(self):
        self._attr = BMAP_DEFAULT_ATTRS
        self._base = None

    def reset(self, **kw):
        self.resetall()
        for k in kw:
            self._attr[k] = kw[k]
    
    def set(self, **kw):
        for k in kw:
            self._attr[k] = kw[k]
    

    def bg(self):
        plt.figure(figsize=self._attr['figsize'], dpi=self._attr['dpi'])
        self._base = Basemap(projection=self._attr['proj'], llcrnrlon = self._attr['lon'][0], llcrnrlat = self._attr['lat'][0], 
                    urcrnrlon = self._attr['lon'][1], urcrnrlat = self._attr['lat'][1], resolution=self._attr['res'])
        self._base.drawcoastlines(color=self._attr['clcolor'], linewidth=self._attr['cllw'])

        if int(self._attr['latinv']):
            self._base.drawparallels(np.arange(self._attr['lat'][0], self._attr['lat'][1], self._attr['latinv']), labels=[1,0,0,0],
                                        color=self._attr['gridc'], linewidth=self._attr['gridlw'], fontsize=self._attr['fontsize'])
        if int(self._attr['loninv']):
            self._base.drawmeridians(np.arange(self._attr['lon'][0], self._attr['lon'][1], self._attr['loninv']), labels=[0,0,0,1],
                                        color=self._attr['gridc'], linewidth=self._attr['gridlw'], fontsize=self._attr['fontsize'])

    def colorbg(self, style=None):
        if style == 'bluemarble':
            self._base.bluemarble(scale=self._attr['clbgs'])
        if style == 'shadedrelief':
            self._base.shadedrelief(scale=self._attr['clbgs'])
        if style == 'etopo':
            self._base.etopo(scale=self._attr['clbgs'])

    def fitboundaries(self, source):
        self._attr['lon'] = [source.long[0,:].min(), source.long[0,:].max()]
        self._attr['lat'] = [source.lat[:,0].min(), source.lat[:,0].max()]
    
    def womesh(self, source):
        _lon, _lat = np.meshgrid(source.long[0,:], source.lat[:,0])
        return self._base(_lon, _lat)

    def quickcountourf(self, source, key):
        self.fitboundaries(source)
        self.bg()
        _X, _Y = self.womesh(source)
        if 'levels' in self._attr.keys():
            self._base.contourf(_X,_Y,source.get(key),cmap=self._attr['cmap'],levels=self._attr['levels'])
        else:
            self._base.contourf(_X,_Y,source.get(key),cmap=self._attr['cmap'])

'''