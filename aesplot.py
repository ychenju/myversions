# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from aesplot_basemap01_beta import *

class settings:

    @staticmethod
    def sans_serif(font : str):
        mpl.rcParams['font.sans-serif']=[font]

class figure:

    def __init__(self, **kwargs):
        self._attr = {}
        self._formats = {}
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]

    def _init_with(self, dic, **kwargs):
        self._attr = {}
        self._formats = {}
        for kw in dic.keys():
            self._attr[kw] = dic[kw]
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]

    def __getitem__(self, key):
        return self._attr[key]

    def __setitem__(self, key, value):
        self._attr[key] = value
        return self._attr[key]

    def format(self, **kwargs):
        for kw in kwargs.keys():
            self._formats[kw] = kwargs[kw]
        self['format'] = True            
        return self

    def plot(self):
        pass

SCATTER_DEFAULT_ATTRS = {

}

class scatter(figure):

    def __init__(self, **kwargs):
        self._init_with(SCATTER_DEFAULT_ATTRS, **kwargs)

    def plot(self):
        if 'x' in self._attr.keys() and 'y' in self._attr.keys():
            self['x'] = np.array(self['x']).reshape(-1)
            self['y'] = np.array(self['y']).reshape(-1)
            if self['x'].shape != self['y'].shape:
                raise RuntimeError('x and y should be in the same shape')
            else:
                for x, y in zip(self['x'],self['y']):
                    plt.plot(x, y, '.', **self._formats)
        elif 'xy' in self._attr.keys():
            for xy in self['xy']:
                plt.plot(*xy, '.', **self._formats)
        else:
            raise RuntimeError('No data')

FUNC_DEFAULT_ATTRS = {
    'xrange': [-10,10],
    'step': 1e-3,
}

class func(figure):
    def __init__(self, **kwargs):
        self._init_with(FUNC_DEFAULT_ATTRS, **kwargs)

    def plot(self):
        if 'f' in self._attr.keys():
            x = np.arange(*self['xrange'], self['step'])
            plt.plot(x, self['f'](x), **self._formats)
        else:
            raise RuntimeError('No data')

FOLDLINE_DEFAULT_ATTRS = {

}

class foldline(figure):

    def __init__(self, **kwargs):
        self._init_with(FOLDLINE_DEFAULT_ATTRS, **kwargs)

    def plot(self):
        if 'x' in self._attr.keys() and 'y' in self._attr.keys():
            plt.plot(self['x'], self['y'], **self._formats)
        else:
            raise RuntimeError('No data')

DOTLINE_DEFAULT_ATTRS = {

}

class dotline(figure):
    def __init__(self, **kwargs):
        self._init_with(DOTLINE_DEFAULT_ATTRS, **kwargs)

    def plot(self):
        if 'x' in self._attr.keys() and 'y' in self._attr.keys():
            self['x'] = np.array(self['x']).reshape(-1)
            self['y'] = np.array(self['y']).reshape(-1)
            if self['x'].shape != self['y'].shape:
                raise RuntimeError('x and y should be in the same shape')
            else:
                plt.plot(self['x'], self['y'], **self._formats)
                for x, y in zip(self['x'],self['y']):
                    plt.plot(x, y, '.', **self._formats)
        else:
            raise RuntimeError('No data')

CONTOURF_DEFAULT_ATTRS = {

}

class contourf(figure):
    def __init__(self, **kwargs):
        self._init_with(CONTOURF_DEFAULT_ATTRS, **kwargs)

    def plot(self):
        if 'x' in self._attr.keys() and 'y' in self._attr.keys() and 'z' in self._attr.keys():
            try:
                plt.contourf(self['x'], self['y'], self['z'], cmap='jet', **self._formats)
            except:
                self['x'] = np.array(self['x']).reshape(-1)
                self['y'] = np.array(self['y']).reshape(-1)
                self['z'] = np.array(self['z']).reshape(-1)
                plt.contourf(self['x'], self['y'], self['z'], cmap='jet', **self._formats)
        else:
            raise RuntimeError('No data')

    def format_levelnumbers(self, x: int):
        self._formats['levels'] = np.arange(np.min(self['z']), np.max(self['z']) + (np.max(self['z'])-np.min(self['z']))/float(x), (np.max(self['z'])-np.min(self['z']))/float(x))
        return self

CONTOUR_DEFAULT_ATTRS = {
    'format': False,
}

class contour(figure):
    def __init__(self, **kwargs):
        self._init_with(CONTOUR_DEFAULT_ATTRS, **kwargs)

    def plot(self):
        if 'x' in self._attr.keys() and 'y' in self._attr.keys() and 'z' in self._attr.keys():
            try:
                plt.contour(self['x'], self['y'], self['z'], cmap='jet', **self._formats)
            except:
                self['x'] = np.array(self['x']).reshape(-1)
                self['y'] = np.array(self['y']).reshape(-1)
                self['z'] = np.array(self['z']).reshape(-1)
                plt.contour(self['x'], self['y'], self['z'], cmap='jet', **self._formats)
        else:
            raise RuntimeError('No data')

    def format_levelnumbers(self, x: int):
        self._formats['levels'] = np.arange(np.min(self['z']), np.max(self['z']) + (np.max(self['z'])-np.min(self['z']))/float(x), (np.max(self['z'])-np.min(self['z']))/float(x))
        return self

AXES_DEFAULT_ATTRS = {
    'format': False,
}

class axes(figure):

    def __init__(self, **kwargs):
        self._init_with(AXES_DEFAULT_ATTRS, **kwargs)

    def plot(self):
        if 'x' in self._attr.keys():
            self['x'] = np.array(self['x']).reshape(-1)
            for x in self['x']:
                plt.axvline(x, **self._formats)
        if 'y' in self._attr.keys():
            self['y'] = np.array(self['y']).reshape(-1)
            for y in self['y']:
                plt.axhline(y, **self._formats)


IMG_DEFAULT_ATTRS = {
    'font': 'default',
    'preset': False,
    'saveas': 'default',
    'serif': 'sans-serif',
}

class image:

    def __init__(self, **kwargs):
        self._attr = {}
        for kw in IMG_DEFAULT_ATTRS.keys():
            self._attr[kw] = IMG_DEFAULT_ATTRS[kw]
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]
        self.preset()

    def __getitem__(self, key):
        return self._attr[key]

    def __setitem__(self, key, value):
        self._attr[key] = value
        return self._attr[key]

    def reset(self, **kwargs):
        self._attr = {}
        for kw in IMG_DEFAULT_ATTRS.keys():
            self._attr[kw] = IMG_DEFAULT_ATTRS[kw]
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]
        self.preset()

    def clear(self):
        plt.clf()

    def creset(self, **kwargs):
        plt.clf()
        self._attr = {}
        for kw in IMG_DEFAULT_ATTRS.keys():
            self._attr[kw] = IMG_DEFAULT_ATTRS[kw]
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]
        self.preset()

    def preset(self, **kwargs):
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]
        if self['font'] != 'default':
            if self['serif'] == 'sans-serif':
                settings.sans_serif(self['font'])
        self['preset'] == True

    def set(self, **kwargs):
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]

    def add(self, fig: figure):
        try:
            fig.plot()
        except:
            print(f'Failed to add the object \'{fig}\'')

    def formatting(self, **kwargs):
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]
        setf.xlim(self)
        setf.ylim(self)
        self.labels()

    def labels(self, **kwargs):
        for kw in kwargs.keys():
            self._attr[kw] = kwargs[kw]
        setf.xlabel(self)
        setf.ylabel(self)
        setf.title(self)

    def save(self):
        try:
            plt.savefig(self['saveas'])
        except:
            raise RuntimeError('Output path (\'saveas\' attr) required.')

    def show(self):
        plt.show()

class setf:

    @staticmethod
    def xlim(img: image):
        if 'xlim' in img._attr.keys():
            plt.xlim(*img['xlim'])

    @staticmethod
    def ylim(img: image):
        if 'ylim' in img._attr.keys():
            plt.ylim(*img['ylim'])

    @staticmethod
    def xlabel(img: image):
        if 'xlabel' in img._attr.keys():
            plt.xlabel(img['xlabel'])

    @staticmethod
    def ylabel(img: image):
        if 'ylabel' in img._attr.keys():
            plt.ylabel(img['ylabel'])

    @staticmethod
    def title(img: image):
        if 'title' in img._attr.keys():
            plt.title(img['title'])
