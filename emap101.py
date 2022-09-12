#import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#import sigmacalc as sc
#import copy

# easy basemap (emap)

EMAP_DEFAULT_ATTRS = {
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
    'intlabels' :   False,          # if the labels of lats and longs are labelled on the integers
}  

class emap:

    def __init__(self, **kw):
        self._attr = {}
        for attr in EMAP_DEFAULT_ATTRS.keys():
            self._attr[attr] = EMAP_DEFAULT_ATTRS[attr]
        self._base = None
        for k in kw:
            self._attr[k] = kw[k]

    def resetall(self):
        self._attr = {}
        for attr in EMAP_DEFAULT_ATTRS.keys():
            self._attr[attr] = EMAP_DEFAULT_ATTRS[attr]
        self._base = None

    def reset(self, **kw):
        self.resetall()
        for k in kw:
            self._attr[k] = kw[k]

    def resettemplate(self, kw):
        self.resetall()
        for k in kw.keys():
            self._attr[k] = kw[k]
    
    def set(self, **kw):
        for k in kw:
            self._attr[k] = kw[k]
    
    def settemplate(self, kw):
        for k in kw.keys():
            self._attr[k] = kw[k]

    def bg(self):
        plt.figure(figsize=self._attr['figsize'], dpi=self._attr['dpi'])
        self._base = Basemap(projection=self._attr['proj'], llcrnrlon = self._attr['lon'][0], llcrnrlat = self._attr['lat'][0], 
                    urcrnrlon = self._attr['lon'][1], urcrnrlat = self._attr['lat'][1], resolution=self._attr['res'])
        self._base.drawcoastlines(color=self._attr['clcolor'], linewidth=self._attr['cllw'])

        if self._attr['intlabels']:
            if int(self._attr['latinv']):
                self._base.drawparallels(np.arange(-90, 90, self._attr['latinv']), labels=[1,0,0,0],
                                            color=self._attr['gridc'], linewidth=self._attr['gridlw'], fontsize=self._attr['fontsize'])
            if int(self._attr['loninv']):
                self._base.drawmeridians(np.arange(-180, 180, self._attr['loninv']), labels=[0,0,0,1],
                                            color=self._attr['gridc'], linewidth=self._attr['gridlw'], fontsize=self._attr['fontsize'])
        else:
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

class emapT(emap):
    def __init__(self, kw):
        self._attr = {}
        for attr in EMAP_DEFAULT_ATTRS.keys():
            self._attr[attr] = EMAP_DEFAULT_ATTRS[attr]
        for k in kw.keys():
            self._attr[k] = kw[k]

class template:
    cres = {'res': 'c'}
    lres = {'res': 'l'}
    ires = {'res': 'i'}
    hres = {'res': 'h'}
    fres = {'res': 'f'}

    @staticmethod
    def res(r: str):
        return {'res': r}

    latinv1_intlbs = {'latinv': 1, 'intlabels': True}
    latinv10_intlbs = {'latinv': 10, 'intlabels': True}
    loninv1_intlbs = {'loninv': 1, 'intlabels': True}
    loninv10_intlbs = {'loninv': 10, 'intlabels': True}

    @staticmethod
    def latinv_intlbs(x):
        return {'latinv': x, 'intlabels': True}
    
    @staticmethod
    def loninv_intlbs(x):
        return {'loninv': x, 'intlabels': True}

    @staticmethod
    def inv_intlbs(x):
        return {'latinv': x, 'loninv': x, 'intlabels': True}

    @staticmethod
    def all(arg, *args):
        d = arg
        for k in args:
            d = dict(d, **k)
        return d

