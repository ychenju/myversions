# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sigmacalc as sc
import copy
import emap as emap

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
        return frame(self.ncfile, *keys)   

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
        self._data = {}    
        self._flag = {}
        for flag in FRAME_DEFAULT_FLAGS.keys():
            self._flag[flag] = FRAME_DEFAULT_FLAGS[flag]
        self._chara = {}   
        self.lat =  copy.deepcopy(np.array(source['XLAT']).squeeze())
        self.long = copy.deepcopy(np.array(source['XLONG']).squeeze())
        self.time = copy.deepcopy(np.array(source['XTIME']).squeeze())
        for key in keys:
            self._data[key] = copy.deepcopy(np.array(source[key]).squeeze())
    
    def load(self, source, *keys):    
        for key in keys:
            self._data[key] = copy.deepcopy(np.array(source.ncfile[key]).squeeze())

    def params(self):   
        _l = [x for x in self._data]
        print(_l)
        return _l
    
    def get(self, key): 
        return self._data[key]
    
    def __getitem__(self, key):
        return self._data[key]
    
    def getall(self):  
        return self._data

    def set(self, key, value):  
        self._data[key] = value
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
        return self._data[key]
    
    def delete(self, key):  
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

    def get3x3(self, key, x, y):
        return np.array([self[key][x+i,y+j] for i in (-1,0,1) for j in (-1,0,1)])

    def mean3x3(self):
        r = voidFrame(self.lat, self.long, self.time)
        for key in self.getall().keys():
            d = []
            for i in range(self[key].shape[0]):
                d.append([])
                for j in range(self[key].shape[1]):
                    d[-1].append(self[key][i,j])

            for i, x in enumerate(d[1:-1]):
                for j, _ in enumerate(x[1:-1]):
                    if np.mean(list(map(filterf.isnan, self.get3x3(key,i,j)))) < 0.5:
                        d[i+1][j+1] = np.nanmean(self.get3x3(key,i,j))
                    else:
                        d[i+1][j+1] = np.nan
            r[key] = np.array(d)
        for flag in self._flag.keys():
            r._flag[flag] = self._flag[flag]
        return r

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
        self._data = {}    
        self._flag = {}
        for flag in FRAME_DEFAULT_FLAGS.keys():
            self._flag[flag] = FRAME_DEFAULT_FLAGS[flag]
        self._chara = {}   
