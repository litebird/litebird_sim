# -*- encoding: utf-8 -*-
#some examples on how to deal with the imo interface
import litebird_sim as lbs

#location of imo flatfile
imoflatfile="/gpfs/work/INF20_lspe/lpagano0/litebird/simteam/litebird_imo/IMOv0.10/"

#load the imo
imo = lbs.Imo(imoflatfile)

#get the scanning parameters
data_file = imo.query("/releases/v0.10/Satellite/scanning_parameters")

print(data_file.metadata)

spin_sun_angle=data_file.metadata['spin_sun_angle_deg']
spin_bore_angle=data_file.metadata['spin_boresight_angle_deg']
print(spin_sun_angle,spin_bore_angle)


#get all detector names for one channel
release="v0.10"
instrument="MFT"
channel="M1-140"
listdet = imo.query("/releases/"+release+"/Satellite/"+instrument+"/"+channel+"/info").metadata['detector_names']

#now loop over them and get the NET and put in a list
NET=list()
for det in listdet:
	NET.append(imo.query("/releases/"+release+"/Satellite/"+instrument+"/"+channel+"/"+det+"/info").metadata['NET_ukrts'])


#loop on all the channels belonging to a given instrument and get total NET of the channel and the average fwhm
release="v0.10"
instrument="LFT"
channels = imo.query("/releases/"+release+"/Satellite/"+instrument+"/info").metadata['channel_names']
for ch in channels:
	data_file = imo.query("/releases/"+release+"/Satellite/"+instrument+"/"+ch+"/info")
	print(ch,data_file.metadata['fwhm_arcmin'],data_file.metadata['NET_channel_ukrts'])

