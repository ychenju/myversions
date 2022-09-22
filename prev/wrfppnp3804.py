# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sigmacalc as sc
import copy
import emap101 as emap

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

    @staticmethod
    def isnotnan(f):
        if np.isnan(f):
            return 0
        else:
            return 1
    

class wrfout:
    ncfile = None

    def __init__(self, path):
        self.ncfile = xr.open_dataset(path)
    
    def extract(self, *keys):
        return frame(self.ncfile, *keys)   

    def __getitem__(self, key):
        return np.array(self.ncfile[key]).squeeze()
    
FRAME_DEFAULT_FLAGS = {
    'CUT': False,
    'LEN':  0,
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
        self.label = '__'
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
        if isnantable(self._data[key]):
            self._chara[key+'_SIGMA'] = np.nan
            return np.nan
        if self._flag['REMOVEWATER']:
            _sig = sc.sigma(self._data['XLONG_RW'], self._data['XLAT_RW'], self._data[key])
        else:
            _sig = sc.sigma(self.long, self.lat, self._data[key])
        self._chara[key+'_SIGMA'] = _sig
        return _sig

    def getsigma(self, key):
        return self._chara[key+'_SIGMA']

    def quickshow(self, key):
        _m = emap.emap()
        _m.quickcountourf(self, key)
        plt.title(key)
        plt.colorbar()
        plt.show()

    def quickshowWithExtensions(self, key, ext):
        _m = emap.emap()
        _m.fitboundaries(ext)
        _m.qcfwpb(self, key)
        plt.title(key)
        plt.colorbar()
        plt.show()

    def showwith(self, key, **kw):
        _m = emap.emap(**kw)
        _m.qcfwpb(self, key)
        plt.title(key)
        plt.colorbar()
        plt.show()

    def showwithfit(self, key, **kw):
        _m = emap.emap(**kw)
        _m.quickcountourf(self, key)
        plt.title(key)
        plt.colorbar()
        plt.show()

    def maxof(self, key):
        return np.array(getflatten(self[key])).max()

    def minof(self, key):
        return np.array(getflatten(self[key])).min()

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
        r.label = self.label + 'MEAN3__'
        return r

    def crop(self, interv=3, fromx=1, tox=-1, fromy=1, toy=-1):
        _r = voidFrame(cp2d(self.lat[fromx:tox:interv, fromy:toy:interv]), cp2d(self.long[fromx:tox:interv, fromy:toy:interv]), self.time)
        for key in self.getall().keys():
            _r[key] = cp2d(self[key][fromx:tox:interv, fromy:toy:interv])
        for flag in self._flag.keys():
            _r._flag[flag] = self._flag[flag]
        _r._flag['RES'] *= interv
        _r.label = self.label + f'CROP:{interv}__'
        return _r

    def lowres3(self, fromx=1, tox=-1, fromy=1, toy=-1):
        r = self.mean3x3()
        r = r.crop(3, fromx=fromx, tox=tox, fromy=fromy, toy=toy)
        return r

    def res(self):
        return self._flag['RES']

    def latrange(self):
        return (self.lat[:,0].min(), self.lat[:,0].max())

    def longrange(self):
        return (self.long[0,:].min(), self.long[0,:].max())

    def anchor(self):
        return self.lat[:,0].min(), self.long[0,:].min()

    def lat1d(self):
        return self.lat[:,0]

    def long1d(self):
        return self.long[0,:]

    def findanchor(self, li: list, lat, lon):
        for l in li:
            if l.anchor() == (lat, lon):
                return l
        return nullFrame()

    def cut(self, interv: int, fromx, fromy):
        fx, fy = int(fromx), int(fromy)
        tx = int(fromx + interv)
        ty = int(fromy + interv)
        r = voidFrame(cp2d(self.lat[fy:ty,fx:tx]), cp2d(self.long[fy:ty,fx:tx]), self.time)
        for key in self.getall().keys():
            r._data[key] = cp2d(self._data[key][fy:ty,fx:tx])
        for flag in self._flag.keys():
            r._flag[flag] = self._flag[flag]
        r.setflag('CUT', True)
        r.setflag('LEN', interv)
        r.label = self.label + f'CUT:{interv}@({fromx},{fromy})__'
        return r
    
    def cutup(self, interv: int): 
        coorlist = []
        for i in range(self.lat.shape[0]//interv):
            for j in range(self.long.shape[1]//interv):
                coorlist.append((i*interv, j*interv))
        r = []
        for coor in coorlist:
            r.append(self.cut(interv, coor[1], coor[0]))
        return r

    def cut3nup(self):
        ilist = []
        while 3**(len(ilist)+1) < np.min(self.lat.shape) and 3**(len(ilist)+1) < np.min(self.long.shape):
            ilist.append(3**(len(ilist)+1))
        r = []
        for interv in ilist:
            r.append(self.cutup(interv))
        return r

def cp2d(attr: np.ndarray):
    r = []
    for i in range(attr.shape[0]):
        r.append([])
        for j in range(attr.shape[1]):
            r[-1].append(attr[i,j])
    return np.array(r)

def correspond(hrdf, lrdf, len, lx, ly):
    thinGrid = hrdf.cut(len*lrdf.res(), lx*lrdf.res(), ly*lrdf.res())
    thickGrid= lrdf.cut(len*hrdf.res(), lx*hrdf.res(), ly*hrdf.res())
    return thinGrid, thickGrid

def getflatten(table: np.ndarray):
    r = []
    for x,y in zip(range(table.shape[0]),range(table.shape[1])):
        r.append(table[y,x])
    return np.array(r)

def isnantable(table: np.ndarray):
    r = np.array(list(filter(filterf.isnotnan, getflatten(table))))
    if len(r) > 0:
        return False
    else:
        return True

def fold(table, length):
    r = []
    for i in range(len(table)//length):
        r.append([])
        for j in range(length):
            r[-1].append(table[i*length+j])
    return np.array(r)

def iserrsigma(sigma, table):
    if sigma > np.array(getflatten(table)).max() - np.array(getflatten(table)).min():
        return True
    else:
        return False

BEAUFORT_SCALE = (  0.2,    1.5,    3.3,    4.5,    7.9,    10.7,   13.8,   17.1,   20.7,   24.4,
                    28.4,   32.6,   36.9,   41.4,   46.1,   50.9,   56.,    61.2)

def beaufort(windspeed):
    if np.isnan(windspeed):
        return np.nan
    for i, x in enumerate(BEAUFORT_SCALE):
        if windspeed <= x:
            return i
    return 18

class voidFrame(frame):
    def __init__(self, lat, long, time):
        self.lat = cp2d(lat)
        self.long = cp2d(long)
        self.time = time
        self._data = {}    
        self._flag = {}
        self.label = '__'
        for flag in FRAME_DEFAULT_FLAGS.keys():
            self._flag[flag] = FRAME_DEFAULT_FLAGS[flag]
        self._chara = {}   

class nullFrame(frame):
    def __init__(self):
        self.lat = None
        self.long = None
        self.time = None
        self._data = {}    
        self._flag = {}
        self.label = 'NULL'
        for flag in FRAME_DEFAULT_FLAGS.keys():
            self._flag[flag] = FRAME_DEFAULT_FLAGS[flag]
        self._chara = {}