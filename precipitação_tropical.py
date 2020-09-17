from   mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
from   netCDF4 import Dataset as Dataset
import numpy as np
import math, copy
import csv
from matplotlib import colors
from numpy import linspace 
from numpy import meshgrid 
import datetime
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
plt.rcParams['axes.xmargin'] = 0
from pyproj import Proj, CRS,transform
import scipy.ndimage as ndimage
import matplotlib.colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage.filters import minimum_filter, maximum_filter
from wrf import getvar, interplevel
import datetime as dt
import cv2
import os
import glob
from PIL import Image


dia='07'
mes= '08'
ano='2020'

t=1
hrs=9
mintxt=30

data_inicial=dt.datetime(int(ano),int(mes),int(dia))

data  = Dataset('/home/lucas/WRF/WRFV3/test/em_real/wrfout_d01_'+ano+'-'+mes+'-'+dia+'_12:00:00')
lats = (data.variables['XLAT'][:,:,:])
lons = (data.variables['XLONG'][:,:,:])
print(lons)
for i in range(192): 
    d=data_inicial+dt.timedelta(hours=hrs,minutes=int(mintxt))
    print('Gerando imagem de precipitação para '+str(d)+'')    
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111)    

    extent = [np.min(lons),np.min(lats), np.max(lons), np.max(lats)]
    min_lon = extent[0]; max_lon = extent[2]; min_lat = extent[1]; max_lat = extent[3]
    bmap = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat, urcrnrlon=max_lon, urcrnrlat=max_lat, epsg=4326)
#    bmap.readshapefile('/home/lucas/Shapefile/pr_municipios/S41MUE250GC_SIR','S41MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')
#    bmap.readshapefile('/home/lucas/Shapefile/ms_municipios/50MUE250GC_SIR','50MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')    
#    bmap.readshapefile('/home/lucas/Shapefile/sc_municipios/42MUE250GC_SIR','42MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')        
#    bmap.readshapefile('/home/lucas/Shapefile/rs_municipios/43MUE250GC_SIR','43MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')            
#    bmap.readshapefile('/home/lucas/Shapefile/sp_municipios/35MUE250GC_SIR','35MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')       
    bmap.readshapefile('/home/lucas/Shapefile/br_unidades_da_federacao/BRUFE250GC_SIR','BRUFE250GC_SIR',linewidth=0.6,color='#C4C4C4')             
    bmap.readshapefile('/home/lucas/Shapefile/ne_10m_admin_0_countries/ne_10m_admin_0_countries','ne_10m_admin_0_countries',linewidth=0.6,color='k')      
    bmap.drawparallels(np.arange( -90., 90.,2.),labels=[1,0,0,0],fontsize=9,linewidth=0,  dashes=[4, 2], color='grey')
    bmap.drawmeridians(np.arange(-180.,180.,2.),labels=[0,0,0,1],fontsize=9,linewidth=0,  dashes=[4, 2], color='grey')       
    
    var = (data.variables['RAINNC'][t,:,:])

    x = linspace(min_lon, max_lon, var.shape[1])
    y = linspace(min_lat, max_lat, var.shape[0])
    x, y = meshgrid(x, y) 

    #caculando a precipitação a cada 30'        
    var2 = (data.variables['RAINNC'][t-1,:,:])
    var=var-var2
    varb = (data.variables['RAINC'][t,:,:])
    var2b = (data.variables['RAINC'][t-1,:,:])
    varb=varb-var2b    
    var=var+varb

    levels=[0,0.1,0.2,0.5,1,1.5,2,2.5,3,4,5,6,8,10,13,16,20,24,28,32]
    cores=['#FFFFFF','#00FB4C','#00E445','#00CD3E','#00B537','#009E2E','#008528','#006F21','#00591A','#FFFF50',
           '#FFD248','#FDA341','#FF763A','#FF2023','#E50C1B','#B80C35','#A00148','#CD0097','#FF00DC'] 
        
    chuva = bmap.contourf(x, y, var, colors=cores, levels=levels)                                                

    #temperatura    
    vart = (data.variables['T2'][t,:,:])
    temp=vart-273.15    
    #calculando a pressão ao nível do mar
    varp =(data.variables['PSFC'][t,:,:])
    p=varp/100
    a =(data.variables['HGT'][t,:,:])    
    base = 1-(0.0065*a/(temp+0.0065*a+273.15 ))
    expoente = -5.257
    slp = p * pow(base,expoente)
    
    #plotando a pressão
    mslp = bmap.contour(x, y, slp, colors='black', linewidths=0.6, levels=np.arange(900,1060,2))
    plt.clabel(mslp, fontsize=7.5, rightside_up=True,inline_spacing=2, fmt="%i")

    ht = getvar(data, "z", units="dm",timeidx=t)
    p = getvar(data, "pressure",timeidx=t)
    hgt500 = interplevel(ht, p, 500.0)
    hgt1000 = interplevel(ht, p, 1000.0) 
    esp = (hgt500-hgt1000)*10

    quente = bmap.contour(x, y, esp, colors='#FF2023', linewidths=0.7, linestyles='dashed',levels=np.arange(5460,6300,60))
    plt.clabel(quente, fontsize=7, rightside_up=True,inline_spacing=2, fmt="%i")
    fria = bmap.contour(x, y, esp, colors='#0305FE', linewidths=0.7,linestyles='dashed', levels=np.arange(3900,5460,60))
    plt.clabel(fria, fontsize=7, rightside_up=True,inline_spacing=2, fmt="%i")
    plt.margins(0,0)   
    #plt.axis('off')

    #plotando altas e baixas pressões
    def extrema(mat,mode='wrap',window=100):
        mn = minimum_filter(mat, size=window, mode=mode)
        mx = maximum_filter(mat, size=window, mode=mode)
        return np.nonzero(mat == mn), np.nonzero(mat == mx)

    local_min, local_max = extrema(slp, mode='wrap', window=100)

    xlows = x[local_min]; xhighs = x[local_max]
    ylows = y[local_min]; yhighs = y[local_max]
    lowvals = slp[local_min]; highvals = slp[local_max]
    m=bmap
    xyplotted = []
    yoffset = 0.013*(m.ymax-m.ymin)
    dmin = yoffset
    for x,y,p in zip(xlows, ylows, lowvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
            dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
            if not dist or min(dist) > dmin:
                baixa=plt.text(x,y,'B',fontsize=17,fontweight='bold', ha='center',va='center',color='#FF2A38')
                valor_b=plt.text(x,y-yoffset,repr(int(p)),fontsize=11, ha='center',va='top',color='#FF2A38')
                baixa.set_path_effects([path_effects.PathPatchEffect(offset=(0., -0.4), hatch='', facecolor='#FF2A38',alpha=0.8), path_effects.PathPatchEffect(edgecolor='#FF2A38', linewidth=0, facecolor='#FF2A38')])    
                valor_b.set_path_effects([path_effects.PathPatchEffect(offset=(0., -0.4), hatch='', facecolor='#FF2A38',alpha=0.8), path_effects.PathPatchEffect(edgecolor='#FF2A38', linewidth=0, facecolor='#FF2A38')])                   
                xyplotted.append((x,y))
    xyplotted = []
    for x,y,p in zip(xhighs, yhighs, highvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
            dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
            if not dist or min(dist) > dmin:
                txtplt=plt.text(x,y,'A',fontsize=17,fontweight='bold',ha='center',va='center',color='#023EFE')
                valor_a=plt.text(x,y-yoffset,repr(int(p)),fontsize=11, ha='center',va='top',color='#023EFE')
                txtplt.set_path_effects([path_effects.PathPatchEffect(offset=(0.1, -0.3), hatch='', facecolor='#023EFE',alpha=0.8), path_effects.PathPatchEffect(edgecolor='#023EFE', linewidth=0, facecolor='#023EFE')])    
                valor_a.set_path_effects([path_effects.PathPatchEffect(offset=(0.1, -0.3), hatch='', facecolor='#023EFE',alpha=0.8), path_effects.PathPatchEffect(edgecolor='#023EFE', linewidth=0, facecolor='#023EFE')])     
                xyplotted.append((x,y))    


    plt.title('\nWRF V4.2 9km: Precipitação (mm/30min), PNMM (hPa), Espessura (1000-500hPa, mgp)\nVálido para: '+str(d)+'', fontweight='bold',fontsize=10,loc='left', va='top')    
    plt.title('\n\nlucasfumagalli@gmail.com', fontsize=12, fontweight='bold', color='#969696',loc='right', va='top')


    cax = fig.add_axes([0.9024, 0.1795, 0.02, 0.6316])
    fig.colorbar(chuva, shrink=.6, cax=cax)
    
	
    if t <10:
       plt.savefig('/home/lucas/Scripts Python/WRF/tropical/rain_out/00'+str(t)+'.png',  
       bbox_inches='tight', dpi=150, transparent=False, pad_inches = 0)
    if t >=10 and t < 100:
       plt.savefig('/home/lucas/Scripts Python/WRF/tropical/rain_out/0'+str(t)+'.png',  
       bbox_inches='tight', dpi=150, transparent=False, pad_inches = 0)       
    if t >=100:
       plt.savefig('/home/lucas/Scripts Python/WRF/tropical/rain_out/'+str(t)+'.png', 
       bbox_inches='tight', dpi=150, transparent=False, pad_inches = 0)
    plt.clf()

    #mapa de temperatura 2m

    print('Gerando imagem de temperatura para '+str(d)+'')    
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(111)    

    extent = [np.min(lons)+1.2,np.min(lats)-0.25, np.max(lons)-1.2, np.max(lats)-0.25]
    min_lon = extent[0]; max_lon = extent[2]; min_lat = extent[1]; max_lat = extent[3]
#    bmap = Basemap(width=10000000,height=8000000, rsphere=(6378137.00,6356752.3142), resolution='f',area_thresh=1000.,projection='lcc', lat_1=max_lat,lat_2=min_lat,lat_0=-28,lon_0=-52)    
    bmap = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat, urcrnrlon=max_lon, urcrnrlat=max_lat, resolution='f', epsg=4326)
#    bmap = Basemap(projection='merc',llcrnrlat=min_lat,urcrnrlat=max_lat,llcrnrlon=min_lon,urcrnrlon=max_lon,lat_ts=-28,resolution='c')        
#    bmap.readshapefile('/home/lucas/Shapefile/pr_municipios/S41MUE250GC_SIR','S41MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')
#    bmap.readshapefile('/home/lucas/Shapefile/ms_municipios/50MUE250GC_SIR','50MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')    
#    bmap.readshapefile('/home/lucas/Shapefile/sc_municipios/42MUE250GC_SIR','42MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')        
#    bmap.readshapefile('/home/lucas/Shapefile/rs_municipios/43MUE250GC_SIR','43MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')            
#    bmap.readshapefile('/home/lucas/Shapefile/sp_municipios/35MUE250GC_SIR','35MUE250GC_SIR',linewidth=0.5,color='#C4C4C4')       
    bmap.readshapefile('/home/lucas/Shapefile/br_unidades_da_federacao/BRUFE250GC_SIR','BRUFE250GC_SIR',linewidth=0.6,color='k')             
    bmap.readshapefile('/home/lucas/Shapefile/ne_10m_admin_0_countries/ne_10m_admin_0_countries','ne_10m_admin_0_countries',linewidth=0.6,color='k')      
    bmap.drawparallels(np.arange( -90., 90.,2.),labels=[1,0,0,0],fontsize=9,linewidth=0,  dashes=[4, 2], color='grey')
    bmap.drawmeridians(np.arange(-180.,180.,2.),labels=[0,0,0,1],fontsize=9,linewidth=0,  dashes=[4, 2], color='grey')       
    
    vart = (data.variables['T2'][t,:,:])
    temp=vart-273.15    

    x = linspace(min_lon, max_lon, temp.shape[1])
    y = linspace(min_lat, max_lat, temp.shape[0])
    x, y = meshgrid(x, y) 


    levels= [-42,-40,-38,-36,-34,-32,-30,-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-9,-8,-7,
             -6,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,
             5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,
             16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,
             25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,32.5,33,33.5,34,34.5,
             35,35.5,36,36.5,37,37.5,38,38.5,39,39.5,40,42,44,46,48,50]
             
    cores= ['#CC3B8C','#CC51A0','#CC66B2','#C578CA','#B36BC0','#9B59B8','#764BA7','#8A64B5','#9884C3','#ADABDC',
    '#BABCE5','#DADAE1','#BABCE5','#ADABDC','#9E98D3','#8187CC','#6B6EC7','#474CB8','#3E63D3','#396DE1',
    '#357BF1','#2997FE','#22A6FE','#1FB1FE','#1ABBFE','#15C7FF','#11D2FF','#0DDEFF','#08E9FF','#04F4FF',
    '#00FFFF','#6FEA9F','#69E494','#60DF8B','#5AD980','#52D375','#4BCE6B','#43C75F','#3BC354','#34BD4B',
    '#2CB840','#25B235','#1EAC2B','#15A71F','#0FA014','#069C0B','#21A411','#33AB1A','#44B221','#54B82B',
    '#66C033','#77C63B','#87CE44','#99D44D','#AADC54','#BAE35E','#CCEA66','#DDF16E','#EDF877','#FFFF80',
    '#FEF67A','#FDEB76','#FDE371','#FBDA6C','#FAD067','#F9C662','#F8BE5E','#F8B559','#F8AB53','#F7A14F',
    '#F39546','#EE8A3F','#EB7E37','#E8722E','#E46627','#DF5A1F','#DC4E18','#D94210','#D43607','#CD1E00',
    '#C61D04','#C11C09','#BA1A0F','#B61913','#AF1819','#AA161E','#9E1328','#98132C','#98132C','#921232',
    '#921232','#9D2539','#A52C40','#AC3447','#B43B4F','#BA4355','#BA4355','#D45C6F','#D45C6F','#E97388',
    '#E97388','#E97388','#E97388','#F186A8','#F186A8','#F797C3','#F797C3','#F7AEEB','#F7AEEB','#FFBCFC',
    '#FFBCFC','#E69FE8','#CB85DC','#9B59B8','#924CAE','#922F93','#8E1475']

    temperatura = plt.contourf(x, y, temp, colors=cores, levels=levels)
#    temperatura2 = plt.contour(x, y, temp, colors='white', linestyles ='solid', linewidths=0.2, levels=np.arange(-42,51,1),alpha=0.3)    
    temperatura3 = plt.contour(x, y, temp, colors='#F34346', linewidths=0.2, levels=np.arange(0,1,1))        

    Latitude,Longitude = [],[]
    with open('cidades.csv') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=',')
        for datai in reader:
            Latitude.append(float(datai['LAT']))
            Longitude.append(float(datai['LON']))

    for ii in range(0,len(Latitude)):
        latslons = [Latitude[ii], Longitude[ii]]     
        plot = {'lat': latslons[0], 'lon': latslons[1]}
        print(plot)
        lat_idx = np.abs(y - plot['lat']).argmin()  
        lon_idx = np.abs(x - plot['lon']).argmin()    
        print(lat_idx)
        print(lon_idx)
        txtplt=plt.text(latslons[1],latslons[0], '{:.0f}'.format(temp[lat_idx, lon_idx]),color='k',ha='center',fontsize=9)



    plt.margins(0,0)   
    #plt.axis('off')

    plt.title('\nWRF V4.2 9km: Temperatura 2m\nVálido para: '+str(d)+'', fontweight='bold',fontsize=10,loc='left', va='top')    
    plt.title('\n\nlucasfumagalli@gmail.com', fontsize=12, fontweight='bold', color='#969696',loc='right', va='top')

    cax = fig.add_axes([0.9024, 0.1795, 0.02, 0.6316])
    fig.colorbar(temperatura, shrink=.6, cax=cax)
    
    resto = t % 2. 

    if resto == 0: 
       mintxt = '30'
       hrs = hrs
    else: 
       mintxt = '00'
       hrs=hrs+1
	
    if t <10:
       plt.savefig('/home/lucas/Scripts Python/WRF/tropical/t2m_out/00'+str(t)+'.png',  
       bbox_inches='tight', dpi=150, transparent=False, pad_inches = 0)
    if t >=10 and t < 100:
       plt.savefig('/home/lucas/Scripts Python/WRF/tropical/t2m_out/0'+str(t)+'.png',  
       bbox_inches='tight', dpi=150, transparent=False, pad_inches = 0)       
    if t >=100:
       plt.savefig('/home/lucas/Scripts Python/WRF/tropical/t2m_out/'+str(t)+'.png', 
       bbox_inches='tight', dpi=150, transparent=False, pad_inches = 0)
    plt.clf()    
    t=t+1
    

image_folder = "/home/lucas/Scripts Python/WRF/tropical/rain_out/*.png"
os.path.getmtime
images = [img for img in sorted(glob.glob(image_folder), key=os.path.realpath)]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('wrf_rain_output.mp4', fourcc, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

image_folder = "/home/lucas/Scripts Python/WRF/tropical/t2m_out/*.png"
os.path.getmtime
images = [img for img in sorted(glob.glob(image_folder), key=os.path.realpath)]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('wrf_t2m_output.mp4', fourcc, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
