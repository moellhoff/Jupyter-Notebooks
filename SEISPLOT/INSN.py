from __future__ import print_function

import sys
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re        # regular expression library

from math import sqrt
from obspy import UTCDateTime
from obspy import read
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from obspy.core.inventory import Inventory

###########################################
# CHECK DATA AVAILABILITY FOR A NETWORK AND GET METADATA

def check_stations_network(net, Year, Month, Day, Hour, Minute, Second):
   # find stations with data
   start_1 = (str(Year)+'-'+str(Month)+'-'+str(Day)+'-'+str(Hour)+'-'+str(Minute)+'-'+str(Second))
   t0 = UTCDateTime(start_1)  
   
   client = Client('IRIS')

   inv = client.get_stations(network = net, level="station")

   network = inv[0] 
   netstat = []
   for i in range(len(network)):
      netstat.extend([net+"."+network[i].code])       

   print(color.GREEN +'data is available for these stations:'+str(netstat)+color.END)

   # now get metadata
   endtime = t0 + 30
   inv = client.get_stations(network=net, station="*", location="*", channel="HHZ", starttime=t0, endtime=endtime,level="response")

   return netstat, t0, inv

##########################################################################
# CHECK DATA AVAILABILITY FOR STATIONS NEAR AN EPICENTER AND GET METADATA

def check_stations(maxradius, SEARCH_TYPE, search_netstat, excl_net, excl_stat, lat, lon, Year, Month, Day, Hour, Minute, Second):
    
   sta_lat = lat; sta_lon = lon
   maxradius = maxradius / 111  # conversion from kn to degrees
   start_1 = (str(Year)+'-'+str(Month)+'-'+str(Day)+'-'+str(Hour)+'-'+str(Minute)+'-'+str(Second))
   UTCDateTime.DEFAULT_PRECISION = 0
   t0 = UTCDateTime(start_1)  

   #################################
   # if only stations in Ireland
   if SEARCH_TYPE == 1:
      sta_lat = 53.522; sta_lon =  -8.744; maxradius = 2.36
   
   ##############################
   # load inventories
   inv = Inventory(networks=[])

   if SEARCH_TYPE != 2:  # if no specific stations are specified
      # Raspberry Shake Network   
      client =  Client(base_url='https://fdsnws.raspberryshakedata.com/')
      try:
         inv += client.get_stations(longitude=sta_lon, latitude=sta_lat, maxradius=maxradius, starttime=t0, endtime=t0+60, channel="EHZ,SHZ",level="channel")
      except:
         print("No Raspberry Shake station data found")
         print('') 
      # BGS network
      client = Client('http://eida.bgs.ac.uk') 
      try:
         inv += client.get_stations(longitude=sta_lon, latitude=sta_lat, maxradius=maxradius, starttime=t0, endtime=t0+60, channel="HHZ",level="channel")
      except:
         print("No GB station data found") 
         print('') 
      # networks provided by GFZ service
      client = Client('GFZ')
      try:
         inv += client.get_stations(longitude=sta_lon, latitude=sta_lat, maxradius=maxradius, starttime=t0, endtime=t0+60, channel="HHZ",level="channel")
      except:
         print("No GFZ station data found")
         print('') 

   if SEARCH_TYPE == 2: # if specific stations are specified
      for stat in search_netstat:
         net = stat.split(".")[0]
         station = stat.split(".")[1]
         # Raspberry Shake Network   
         if net == 'AM':
            client =  Client(base_url='https://fdsnws.raspberryshakedata.com/')
            try:
               inv += client.get_stations(network=net, station=station, starttime=t0, endtime=t0+60, channel="EHZ,SHZ",level="channel")
            except:
               print("No Raspberry Shake station data found")
               print('') 
         # BGS network
         if net == 'GB':
            client = Client('http://eida.bgs.ac.uk')
            try:
               inv += client.get_stations(network=net, station=station, starttime=t0, endtime=t0+60, channel="HHZ",level="channel")
            except:
               print("No GB station data found")
               print('') 
         # networks provided by GFZ service
         if net != 'AM' and net != 'GB':
            client = Client('GFZ')
            try:
               inv += client.get_stations(network=net, station=station, starttime=t0, endtime=t0+60, channel="HHZ",level="channel")
            except:
               print("No GFZ station data found")
               print('') 

   ########################################
   # populate station entries nslc (network station location channel)
   nslc = []
   for network in inv:
       for station in network:
          for channel in station:
             nslc.extend([network.code+"."+station.code+"."+channel.location_code+"."+channel.code])  

   # remove multiple entries:        
   nslc = list(dict.fromkeys(nslc))

   print('found data for these stations in provided search area and time frame:')             
   print(nslc)

   ###################################
   # exclude certain networks/stations 

   excl_netstat = []
   for item in nslc:
      for ex_n in excl_net:
         if re.search(ex_n+'.+', item):  # The .+ symbol is used in place of * symbol
            excl_netstat.append(item)               
   for item in excl_netstat: 
      nslc.remove(item)  

   excl_netstat = []  
   for item in nslc:
      for ex_s in excl_stat:
         if re.search('.+'+ex_s+'.+', item):  # The .+ symbol is used in place of * symbol
            excl_netstat.append(item)
   for item in excl_netstat: 
      nslc.remove(item)             

   print('')             
   print('new list after excluding networks and/or stations:')             
   print(nslc)
   #print(inv)
   
   return nslc, t0, inv

#############################################
# SORT STATIONS BY DISTANCE FROM EPICENTER

def sort_stations(nslc, t0, inv, lat, lon, depth):

   R_hypos = []
   no_st = len(nslc)
   for i in range(no_st):       
      station = nslc[i]
      station_coord = inv.get_coordinates(nslc[i],t0) 
      R_epi = 0.001*gps2dist_azimuth(station_coord["latitude"], station_coord["longitude"], float(lat), float(lon))[0]
      R_hypo = sqrt(R_epi**2 + (float(depth))**2)
      R_hypos.extend([R_hypo])

   # sort stations depending on hypocentral distance
   s = np.array(nslc)
   R = np.array(R_hypos)
   inds = R.argsort()
   nslc = s[inds]
   R_hypos = R[inds]

   print(nslc)
   return nslc, R_hypos

#################################################
# PLOT SEISMOGRAMS

def plot(nslc, correct, t0, lat, lon, length, pretime, R_hypos, freqmin, freqmax):

   if (correct != 'counts' and correct != 'disp' and correct != 'vel' and correct != 'acc'):
      sys.exit('No valid amplitude parameter specified, exiting program now.')
      
   no_st = len(nslc)
   plt.style.use("default") # or in jupyter get grey background
#   plt.style.use("classic") # or in jupyter get grey background
   plt.figure(figsize=(10,no_st*2+1),dpi=75)

   if freqmin == 'none' and freqmax == 'none':
      filter = 'none'
   if freqmin != 'none' and freqmax == 'none':
      filter = 'HP'
   if freqmin == 'none' and freqmax != 'none':
      filter = 'LP'
   if freqmin != 'none' and freqmax != 'none':
      filter = 'BP'

   for i in range(no_st):       
      str_nslc=str(nslc[i])
      net = str_nslc.split(".")[0]
      station = str_nslc.split(".")[1]
      location = str_nslc.split(".")[2]
      channel = str_nslc.split(".")[3]
      R_hypo = R_hypos[i]
      print(net, station, "{:.0f}".format(round(R_hypo,2)), "km")

      if net != 'AM' and net != 'GB':
#         client = Client('IRIS')
         client = Client('GFZ')
      elif net == 'AM':
#         client = Client(base_url='https://fdsnws.raspberryshakedata.com/')
         client = Client('https://fdsnws.raspberryshakedata.com')
      elif net == 'GB':
         client = Client('http://eida.bgs.ac.uk')
      
      # 60 extra seconds in case of filtering...  
      st = client.get_waveforms(net, station, location, channel, t0-60, t0 + length +60, attach_response=True)       

      # instrument correction
      if correct == 'disp':
         st.remove_response(output="DISP")
      if correct == 'vel':
         st.remove_response(output="VEL")
      if correct == 'acc':
         st.remove_response(output="ACC")
       
      if filter == 'HP':
         st.filter('highpass', freq=freqmin, corners=4, zerophase=True)
      if filter == 'LP':
         st.filter('lowpass', freq=freqmax, corners=4, zerophase=True)
      if filter == 'BP':
         st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

      #st.slice(t0, t0 + length) 
      tr = st.slice(t0-pretime, t0 + length)[0]
      if (correct == 'disp' or correct != 'vel' or correct != 'acc'):
         tr.data *= 1e6 # plot in units of micro 

      t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
      t = t - pretime
      #convert from seconds to minutes:
      t = t / 60.0
      i=i+1
      plt.subplot(no_st,1,i)
      x1 = plt.gca()

      plt.plot(t, tr.data, 'k', label='%s km' %(str(int(round(R_hypo)))))
      # add legend
      legend = x1.legend(loc='upper right')
      # remove line in legend + use small font
      leg = x1.legend(handlelength=0, handletextpad=0, fancybox=True, fontsize='small')
      for item in leg.legendHandles:
         item.set_visible(False)
      plt.grid()
      # x- and y-limits
      (ymin, ymax) = plt.ylim()
      #ymax_abs = max(-ymin,ymax)
      #plt.ylim(-ymax_abs, ymax_abs)
      plt.xlim(-pretime/60., (length-pretime)/60.)

      if i < no_st:
         x1.axes.xaxis.set_ticklabels([])   
      if i == no_st:   
         plt.xlabel('time [minutes since origin time]')
      #plt.xlim((t0-60)/60, (t0 + length)/60)

      if correct == 'counts':
         plt.ylabel('amplitude [counts]')        
      if correct == 'disp':
         plt.ylabel('amplitude ['+r'$\mu m}$'+']')
      if correct == 'vel':
         plt.ylabel('amplitude ['+r'$\mu m/s$'+']')
      if correct == 'acc':
         plt.ylabel('amplitude ['+r'$\mu m/s^{2}$'+']')

      if filter == 'HP':
         str_filter = str(freqmin)+"Hz HP"
      if filter == 'LP':
         str_filter = str(freqmax)+"Hz LP"
      if filter == 'BP':
         str_filter = str(freqmin)+"Hz - "+str(freqmax)+"Hz"
#      plt.title('%s.%s.%s.%s    %s' %(net,station,location,channel,str_filter),loc='right')
      plt.title('%s.%s    %s' %(net,station,str_filter),loc='right')
      plt.plot((0, 0), (ymin, ymax), 'r-') # time of earthquake
      #######################################################################################
      # plot titles
      if i == 1:
         plt.title('%s, %s, %s' %(t0.strftime("%d.%m.%Y, %H:%M:%S"), lat, lon))

      place = ''
      if net == 'AM':
         place = 'Citizen Station (Rasp. Shake)'
      if station == 'DSB':
         place = 'Dublin, Ireland (DIAS)'
      if station == 'IDGL':
         place = 'Donegal, Ireland (DIAS)'
      if station == 'IGLA':
         place = 'Galway, Ireland (DIAS)'
      if station == 'ILTH':
         place = 'Louth, Ireland (DIAS)'
      if station == 'IWEX':
         place = 'Wexford, Ireland (DIAS)'
      if station == 'VAL':
         place = 'Kerry, Ireland (DIAS)'
      if station == 'LEWI':
         place = ' Isle of Lewis, Scotland (BGS)'
      if station == 'KPL':
         place = ' Plockton, Scotland (BGS)'
      if station == 'LAWE':
         place = ' Loch Awe, Scotland (BGS)'
      if station == 'PGB1':
         place = ' Glasgow, Scotland (BGS)'
      if station == 'CLGH':
         place = ' Antrim, Northern Ireland (BGS)'
      if station == 'NEWG':
         place = ' New Galloway, Scotland (BGS)'
      if station == 'GAL1':
         place = ' Galloway, Scotland (BGS)'
      if station == 'ESK':
         place = ' Eskdalemuir, Scotland (BGS)'
      if station == 'KESW':
         place = ' Keswick, England (BGS)'
      if station == 'IOMK':
         place = 'Isle of Man (BGS)'
      if station == 'WLF1':
         place = 'Anglesey, Wales (BGS)'
      if station == 'WPS':
         place = 'Anglesey, Wales (BGS)'
      if station == 'FOEL':
         place = ' Llangollen, Wales (BGS)'
      if station == 'HLM1':
         place = ' Shropshire, England (BGS)'
      if station == 'RSBS':
         place = 'Pembrokeshire, Wales (BGS)'
      if station == 'HTL':
         place = ' Devon, England (BGS)'
      if station == 'CCA1':
         place = ' Cornwall, England (BGS)'
      if station == 'JSA':
         place = ' Jersey, Channel Islands (BGS)'
      if station == 'ROSA':
         place = ' Azores, Portugal'
      plt.title('%s' %(place),loc='left')
  


   plt.tight_layout()



