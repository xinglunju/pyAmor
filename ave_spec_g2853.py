import os, aplpy, time, csv, sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.convolution import *
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import splrep,splev
from lmfit import minimize, Parameters, report_fit

start = time.clock()
print 'Start the timer...'

def __nh3_init__():
	"""
	nh3_init()
	Initialize the parameters. Call with no keys.
	"""
	nh3_info = {}
	nh3_info['energy'] = [23.4, 64.9, 125., 202., 298., 412.] # Upper level energy (Kelvin)
	nh3_info['frest'] = [23.6944955, 23.7226333, 23.8701292, \
	                     24.1394163, 24.5329887, 25.0560250] # Rest frequency (GHz)
	nh3_info['c0'] = 1.89e5 # Constants needed in calculation of
	nh3_info['b0'] = 2.98e5 # partition function
	nh3_info['Ijk'] = [50.0,  79.6,  89.4,  93.5,  95.6,  96.9] # Ratio of the main line intensity to all hyperfine
	nh3_info['lratio'] = [0.278, 0.222, 0.0653, 0.0302, 0.0171, 0.0115, 0.0083] # Ratio of the inner hyperfine intensity relative to the main line
	nh3_info['vsep'] = [7.756, 19.372, 16.57, 21.49, 24.23, 25.92, 26.94] # velocity separation between the mainline and the inner satellites (and outer satellites for (1,1)) (km/s)
	return nh3_info

def __nh3_read__(fitsfile):
	"""
	nh3_read(fitsfile,meta,hdr)
	Load the FITS image, save the metadata and the header.
	"""
	try:
		img = fits.open(fitsfile)
	except IOError:
		return False
	meta = img[0].data
	hdr = img[0].header
	if hdr['naxis'] == 4:
		meta = meta[0,:,:,:]
		hdr.remove('naxis4')
		hdr.remove('crpix4')
		hdr.remove('cdelt4')
		hdr.remove('crval4')
		hdr.remove('ctype4')
		hdr['naxis'] = 3
		print 'Extra axis removed. NH3 file sucessfully loaded.'
	elif hdr['naxis'] == 3:
		print 'NH3 file sucessfully loaded.'
	else:
		print 'The file is not a cube. Find the correct one please!'
	return meta,hdr
	img.close()

def __nh3_load_axes__(header):
	"""
	nh3_load_axes(header)
	Load the axes based on the header.
	"""
	naxisx = header['naxis1']
	naxisy = header['naxis2']
	naxisv = header['naxis3']
	xarray = np.arange(0,naxisx,1)
	yarray = np.arange(0,naxisy,1)
	w = wcs.WCS(header)
	# X axis
	y_mid = np.round(naxisy/2) - 1.0
	y_mid = [y_mid] * naxisx
	ra_pix = np.vstack([xarray,y_mid,np.zeros(naxisx)])
	ra_pix = ra_pix.transpose()
	print ra_pix.shape
	xaxis = w.wcs_pix2world(ra_pix,1)
	xaxis = xaxis[:,0]
	# Y axis
	x_mid = np.round(naxisx/2) - 1.0
	x_mid = [x_mid] * naxisy
	dec_pix = np.vstack([x_mid,yarray,np.zeros(naxisy)])
	dec_pix = dec_pix.transpose()
	yaxis = w.wcs_pix2world(dec_pix,1)
	yaxis = yaxis[:,1]
	# V axis
	onevpix = header['cdelt3']*0.001
	v0 = header['crval3']*0.001
	v0pix = int(header['crpix3'])
	vaxis = onevpix * (np.arange(naxisv)+1-v0pix) + v0
	return xaxis, yaxis, vaxis

def __interp_upp2low__(data_upp):
	"""
	interp_upp2low(data_upp,xaxis_upp,)
	Interpolate the cube of the upper level to that of the lower level.
	"""
	delta = np.radians(hdr2['crval2'])
	onexpix_prj = onexpix/np.cos(delta)
	x0_upp = xaxis_upp[0]
	y0_upp = yaxis_upp[0]
	v0_upp = vaxis_upp[0]

	xaxis = (xaxis_low - x0_upp)/onexpix_prj
	yaxis = (yaxis_low - y0_upp)/oneypix
	vaxis = (vaxis_low - v0_upp)/onevpix

	Y, V, X = np.meshgrid(yaxis,vaxis,xaxis)
	data_upp = map_coordinates(data_upp,(V,Y,X),order=1)
	return data_upp
	print 'Interpolate the cube of the upper level to the lower'

def __xcorrelate__(spec, vaxis):
	"""
	__xcorrelate(spec)
	Return the channel index of the VLSR, through a cross-correlation between input spectrum
	and a model.
	"""
	voff_lines = [7.56923, 0.0, -7.47385]
	tau_wts = [0.278, 1.0, 0.278]
	vlength = len(vaxis)
	stretch = 10.0
	oldidx = np.linspace(0,vlength-1,vlength)
	newidx = np.linspace(0,vlength-1,vlength*stretch)
	temp = splrep(oldidx, vaxis)
	newvaxis = splev(newidx,temp)
	temp = splrep(oldidx, spec)
	newspec = splev(newidx,temp)
	kernel = np.zeros(vlength*10)
	sigmav = 1.0
	for i in np.arange(len(voff_lines)):
		kernel += np.exp(-(newvaxis-newvaxis[vlength*stretch/2]-voff_lines[i])**2/(2*sigmav**2))*tau_wts[i]
	lags = np.correlate(newspec,kernel,'same')
	vlsr = newvaxis[lags.argmax()]
	return vlsr

def __loadvlsr__(infile, gbtfits):
	"""
	__loadvlsr(infile, gbtfits)
	Find the VLSR based on Zoey's result, return an array which contains RA, DEC, first VLSR, 
	second VLSR of each pixel.
	"""
	# Load the GBT NH3 FITS image
	data_gbt, hdr_gbt = __nh3_read__(gbtfits)
	xaxis, yaxis, vaxis = __nh3_load_axes__(hdr_gbt)

	firstguess = np.zeros([4,hdr_gbt['NAXIS2'],hdr_gbt['NAXIS1']])

	# Open Zoey's result
	temp = open(infile)
	text = temp.readlines()
	for block in np.arange(len(text)/14):
		indices = text[block*14].split()
		xno = np.int(indices[0])
		yno = np.int(indices[1])
		firstguess[0,yno,xno] = xaxis[xno]
		firstguess[1,yno,xno] = yaxis[yno]
		vhit1 = np.round(np.float(text[block*14+1].split()[2]))
		vhit2 = np.round(np.float(text[block*14+6].split()[2]))
		if vhit1 > 0:
			firstguess[2,yno,xno] = vaxis[vhit1]
		if vhit2 > 0:
			firstguess[3,yno,xno] = vaxis[vhit2]
	temp.close()
	del temp

	return firstguess, xaxis, yaxis

def __gauss_tau__(axis,p):
	"""
	Genenerate a Gaussian model given an axis and a set of parameters.
	Params:[peaki, tau11, peakv, sigmav]
	"""
	sx = len(axis)
	u  = ((axis-p[2])/np.abs(p[3]))**2
	f = -1.0 * p[1] * np.exp(-0.5*u)
	f = -1.0 * p[0] * np.expm1(f)
	return f

def __model_11__(params, vaxis, spec):
	"""
	Model three components of NH3 (1,1), subtract data.
	"""
	peaki = params['peaki'].value
	sigmav = params['sigmav'].value
	peakv = params['peakv'].value
	tau11 = params['tau11'].value
	peaki_s1 = params['peaki_s1'].value
	sigmav_s1 = params['sigmav_s1'].value
	peakv_s1 = params['peakv_s1'].value
	tau11_s1 = params['tau11_s1'].value
	peaki_s2 = params['peaki_s2'].value
	sigmav_s2 = params['sigmav_s2'].value
	peakv_s2 = params['peakv_s2'].value
	tau11_s2 = params['tau11_s2'].value
	peaki_upp = params['peaki_upp'].value
	sigmav_upp = params['sigmav_upp'].value
	peakv_upp = params['peakv_upp'].value
	tau22 = params['tau22'].value
	model = __gauss_tau__(vaxis,[peaki,tau11,peakv,sigmav]) + \
			__gauss_tau__(vaxis,[peaki_s1,tau11_s1,peakv_s1,sigmav_s1]) + \
			__gauss_tau__(vaxis,[peaki_s2,tau11_s2,peakv_s2,sigmav_s2]) + \
			__gauss_tau__(vaxis,[peaki_upp,tau22,peakv_upp,sigmav_upp])
	return model - spec

def __model_11_2c__(params, vaxis, spec):
	"""
	Model three components of NH3 (1,1), subtract data.
	"""
	temp = __model_11__(params, vaxis, spec)
	peaki_c2 = params['peaki_c2'].value
	sigmav_c2 = params['sigmav_c2'].value
	peakv_c2 = params['peakv_c2'].value
	tau11_c2 = params['tau11_c2'].value
	peaki_s1_c2 = params['peaki_s1_c2'].value
	sigmav_s1_c2 = params['sigmav_s1_c2'].value
	peakv_s1_c2 = params['peakv_s1_c2'].value
	tau11_s1_c2 = params['tau11_s1_c2'].value
	peaki_s2_c2 = params['peaki_s2_c2'].value
	sigmav_s2_c2 = params['sigmav_s2_c2'].value
	peakv_s2_c2 = params['peakv_s2_c2'].value
	tau11_s2_c2 = params['tau11_s2_c2'].value
	peaki_upp_c2 = params['peaki_upp_c2'].value
	sigmav_upp_c2 = params['sigmav_upp_c2'].value
	peakv_upp_c2 = params['peakv_upp_c2'].value
	tau22_c2 = params['tau22_c2'].value
	model = __gauss_tau__(vaxis,[peaki_c2,tau11_c2,peakv_c2,sigmav_c2]) + \
			__gauss_tau__(vaxis,[peaki_s1_c2,tau11_s1_c2,peakv_s1_c2,sigmav_s1_c2]) + \
			__gauss_tau__(vaxis,[peaki_s2_c2,tau11_s2_c2,peakv_s2_c2,sigmav_s2_c2]) + \
			__gauss_tau__(vaxis,[peaki_upp_c2,tau22_c2,peakv_upp_c2,sigmav_upp_c2])
	return temp + model

def __tau_11__(mainflux,sateflux,cutoff=0.008):
	"""
	nh3_tau_low(model)
	Calculate the optical depth of the main hyperfine of NH3 (1,1).
	"""
	tau = 0
	taumax = 10.0
	taumin = 0.001
	aconst_in = nh3_info['lratio'][0]
	t = np.arange(5000) * (taumax - taumin)/5000. + taumin
	rat_in = (1-np.exp(-t))/(1-np.exp(-aconst_in*t))
	print 'Ratios between ', rat_in[4999], ' and ', rat_in[0], ' are valid.'

	if mainflux >= cutoff:
		tau = taumin
		if sateflux >= cutoff:
			diff = 9999.
			best = 0.
			ratio = mainflux/sateflux
			best = np.abs(ratio - rat_in).argmin()
			tau = t[best]
	return tau

clickvalue = []
def onclick(event):
	print 'The Vlsr you select: %f' % event.xdata
	clickvalue.append(event.xdata)
	#clickvalue = clickvalue + [event.xdata]

def fit_2comp(data_low, data_upp, ra_range=[0,1], dec_range=[0,1], cutoff=0.010, varyv=2, writefits=False, interactive=False, mode='single'):
	"""
	fits_2comp(data_low, data_upp)
	Use Monte Carlo approach to derive the temperature based on two transitions.
	Also locate two velocity components, if there are any, and derive temperatures respectively.
	The velocity components are pre-identified based on GBT data by Zoey.
	"""
	trot = np.zeros([naxisy,naxisx])
	trot_error = np.zeros([naxisy,naxisx])
	linew11 = np.zeros([naxisy,naxisx])
	linew22 = np.zeros([naxisy,naxisx])
	linewratio = np.zeros([naxisy,naxisx])
	peakv = np.zeros([naxisy,naxisx])
	#tauupp = np.zeros([naxisy,naxisx])

	#lookup = fits.open('intrinsiclw_lookup.fits')
	#metalu = lookup[0].data
	#hdrlu = lookup[0].header
	#crval1 = hdrlu['CRVAL1']
	#cdelt1 = hdrlu['CDELT1']
	#crpix1 = hdrlu['CRPIX1']
	#crval2 = hdrlu['CRVAL2']
	#cdelt2 = hdrlu['CDELT2']
	#crpix2 = hdrlu['CRPIX2']

	if interactive:
		plt.ion()
		f = plt.figure(figsize=(14,8))
		ax = f.add_subplot(111)

	for i in dec_range:
		for j in ra_range:
			print 'Start to fit pixel (%d,%d)' % (j,i)
			spec_low = data_low[:,i,j]
			spec_low[0:10] = 0.0
			spec_low[-10:-1] = 0.0
			spec_low[-1] = 0.0
			spec_upp = data_upp[:,i,j]
			spec_upp[0:20] = 0.0
			spec_upp[-20:-1] = 0.0
			spec_upp[-1] = 0.0
			spec = np.concatenate((spec_low, spec_upp))
			#spec[np.where(spec<=cutoff)] = 0.0
			vaxis = np.concatenate((vaxis_low, vaxis_low+40.0))
			#xval = xaxis_low[j]
			#yval = yaxis_low[i]
			#x_hitpix = np.abs(xaxis_gbt-xval).argmin()
			#y_hitpix = np.abs(yaxis_gbt-yval).argmin()
			#vlsr1 = firstguess[2,y_hitpix,x_hitpix]
			#vlsr2 = firstguess[3,y_hitpix,x_hitpix]
			#print 'The two velocity components are:', vlsr1, vlsr2
			#vlsr1 = 86.5
			#vlsr2 = 88.5
			

			#if interactive:
				#plt.plot(vaxis, spec, 'k+', label='Original')
				#plt.legend()
				#plt.show()

			unsatisfied = True
			while unsatisfied:
				if interactive:
					f.clear()
					plt.plot(vaxis, spec, 'k+', label='Original')
					cutoff_line = [cutoff] * len(vaxis)
					cutoff_line_minus = [-1.0*cutoff] * len(vaxis)
					plt.plot(vaxis, cutoff_line, 'r-')
					plt.plot(vaxis, cutoff_line_minus, 'r-')
					#clickvalue = []
					if mode == 'single':
						cid = f.canvas.mpl_connect('button_press_event', onclick)
						raw_input('Click on the plot to select Vlsr...')
						print clickvalue
						if len(clickvalue) >= 1:
							print 'Please select at least one velocity! The last one will be used.'
							vlsr1 = clickvalue[-1]
						elif len(clickvalue) == 0:
							vlsr1 = 0.0
						print 'The Vlsr is %0.2f' % vlsr1
						raw_input('Press any key to start fitting...')
						f.canvas.mpl_disconnect(cid)
						vlsr2 = 0.0
					elif mode == 'double':
						cid = f.canvas.mpl_connect('button_press_event', onclick)
						raw_input('Click on the plot to select Vlsrs...')
						print clickvalue
						if len(clickvalue) >= 2:
							print 'Please select at least two velocities! The last two will be used.'
							vlsr1,vlsr2 = clickvalue[-2],clickvalue[-1]
						elif len(clickvalue) == 1:
							vlsr1 = clickvalue[-1]
							vlsr2 = 0.0
						elif len(clickvalue) == 0:
							vlsr1,vlsr2 = 0.0,0.0
						print 'Or input two velocities manually:'
						manualv = raw_input()
						manualv = manualv.split()
						if len(manualv) == 2:
							vlsr1,vlsr2 = np.float_(manualv)
						else:
							print 'Invalid input...'
						print 'The two Vlsrs are %0.2f km/s and %0.2f km/s.' % (vlsr1,vlsr2)
						raw_input('Press any key to start fitting...')
						f.canvas.mpl_disconnect(cid)
					else:
						vlsr1,vlsr2 = 0.0,0.0
				else:
					if mode == 'single':
						if spec_low.max() >= cutoff:
							vlsr1 = __xcorrelate__(spec_low, vaxis_low)
							if vlsr1 <=82 or vlsr1 >=92:
								vlsr1 = 0.0
							if spec_low[np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
								vlsr1 = 0.0
							if spec_low[np.abs(vaxis_low - vlsr1 + 7.47385).argmin()] <= cutoff and spec_low[np.abs(vaxis_low - vlsr1 - 7.56923).argmin()] <= cutoff:
								vlsr1 = 0.0
							if spec_upp[np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
								vlsr1 = 0.0
						else:
							vlsr1 = 0.0
						vlsr2 = 0.0
					elif mode == 'double':
						vlsr1,vlsr2 = 86.0,88.0
					else:
						vlsr1,vlsr2 = 0.0,0.0


				# 17 parameters, but only 7 are indenpendent 
				params = Parameters()
				if vlsr1 != 0:
					params.add('peaki', value=0.030, min=0, max=0.050)
					params.add('tau11', value=1.0, min=0, max=10.0)
					if varyv > 0:
						params.add('peakv', value=vlsr1, min=vlsr1-varyv*onevpix, max=vlsr1+varyv*onevpix)
					elif varyv == 0:
						params.add('peakv', value=vlsr1, vary=False)
					params.add('sigmav', value=1.0, min=0, max=5.0)
					params.add('peaki_s1', expr="peaki*(1-exp(-tau11_s1))/(1-exp(-tau11))")
					params.add('tau11_s1', expr='tau11*0.278')
					params.add('peakv_s1', expr='peakv-7.47385')
					params.add('sigmav_s1', expr='sigmav')
					params.add('peaki_s2', expr='peaki_s1')
					params.add('tau11_s2', expr='tau11_s1')
					params.add('peakv_s2', expr='peakv+7.56923')
					params.add('sigmav_s2', expr='sigmav')
					params.add('peaki_upp', expr='peaki*(1-exp(-tau22))/(1-exp(-tau11))', min=0, max=0.050)
					#params.add('tau22', value=1.0, min=0, expr='<=tau11/0.282')
					params.add('tau22', value=1.0, min=0, max=10.0)
					params.add('peakv_upp', expr='peakv+40.0')
					#params.add('sigmav_upp', value=1.0, min=0, max=5.0)
					params.add('sigmav_upp', expr='sigmav')
					params.add('Trot', value=0.0, expr='-41.5/log(0.282*tau22/tau11)', min=0)
				# another 17 parameters for the second component
				if vlsr2 != 0:
					params.add('peaki_c2', value=0.030,  min=0, max=0.050)
					params.add('tau11_c2', value=1.0, min=0, max=10.0)
					if varyv > 0:
						params.add('peakv_c2', value=vlsr2, min=vlsr2-varyv*onevpix, max=vlsr2+varyv*onevpix)
					elif varyv == 0:
						params.add('peakv_c2', value=vlsr2, vary=False)
					params.add('sigmav_c2', value=1.0, min=0, max=5.0)
					params.add('peaki_s1_c2', expr="peaki_c2*(1-exp(-tau11_s1_c2))/(1-exp(-tau11_c2))")
					params.add('tau11_s1_c2', expr='tau11_c2*0.278')
					params.add('peakv_s1_c2', expr='peakv_c2-7.47385')
					params.add('sigmav_s1_c2', expr='sigmav_c2')
					params.add('peaki_s2_c2', expr='peaki_s1_c2')
					params.add('tau11_s2_c2', expr='tau11_s1_c2')
					params.add('peakv_s2_c2', expr='peakv_c2+7.56923')
					params.add('sigmav_s2_c2', expr='sigmav_c2')
					params.add('peaki_upp_c2', expr='peaki_c2*(1-exp(-tau22_c2))/(1-exp(-tau11_c2))', min=0, max=0.050)
					#params.add('tau22_c2', value=1.0, min=0, expr='<=tau11_c2/0.282')
					params.add('tau22_c2', value=1.0, min=0, max=10.0)
					params.add('peakv_upp_c2', expr='peakv_c2+40.0')
					#params.add('sigmav_upp_c2', value=1.0, min=0, max=5.0)
					params.add('sigmav_upp_c2', expr="sigmav_c2")
					params.add('Trot_c2', value=0.0, expr='-41.5/log(0.282*tau22_c2/tau11_c2)', min=0)

				# do fit, here with leastsq model
				#spec[np.where(spec<=cutoff)] = 0.0
				if vlsr1 != 0 and vlsr2 != 0:
					result = minimize(__model_11_2c__, params, args=(vaxis, spec))
				elif vlsr1 != 0 or vlsr2 != 0:
					result = minimize(__model_11__, params, args=(vaxis, spec))
				else:
					unsatisfied = False
					continue
				
				final = spec + result.residual
				report_fit(params)

				if interactive:
					plt.plot(vaxis, final, 'r', label='Fitting result')
					if vlsr1 != 0 and vlsr2 != 0:
						final_c1 = __model_11__(params, vaxis, spec) + spec
						final_c2 = final - final_c1
						plt.plot(vaxis, final_c1, 'm--', label='1st component', linewidth=2)
						plt.plot(vaxis, final_c2, 'c--', label='2nd component', linewidth=2)
						plt.text(0.05, 0.9, '1st Trot=%.1f'%params['Trot'].value, transform=ax.transAxes, color='m')
						plt.text(0.05, 0.8, '2nd Trot=%.1f'%params['Trot_c2'].value, transform=ax.transAxes, color='c')
					elif vlsr1 != 0 or vlsr2 != 0:
						plt.text(0.05, 0.9, 'Trot=%.1f'%params['Trot'].value, transform=ax.transAxes, color='r')
					plt.legend()
					plt.show()
					print 'Is the fitting ok? y/n'
					yn = raw_input()
					if yn == 'y':
						unsatisfied = False 
					else:
						unsatisfied = True
					#raw_input('Press any key to continue...')
					f.clear()
				else:
					unsatisfied = False
					f.clear()

				if writefits:
					# write the temperature
					trot[i,j] = params['Trot'].value
					trot[np.where(trot>50.)] = 50.
					trot[np.where(trot<0.)] = 0.
					trot_error[i,j] =  params['Trot'].stderr
					trot_error[np.where(trot_error>10.)] = 10.
					trot_error[np.where(trot_error<0.)] = 0.
					linew11[i,j] = params['sigmav'].value
					linew22[i,j] = params['sigmav_upp'].value
					#linewratio[i,j] = params['sigmav_upp'].value / params['sigmav'].value
					peakv[i,j] = params['peakv'].value

					#col_index = np.argmin(params['tau11'].value)
					#linew_intrinsic = (np.where(linew11))

	if writefits:
		hdrt = hdr1
		hdrt.remove('naxis3')
		hdrt.remove('crpix3')
		hdrt.remove('cdelt3')
		hdrt.remove('crval3')
		hdrt.remove('ctype3')
		hdrt['naxis'] = 2
		hdrt['bunit'] = 'K'
		trot[np.where(trot>=50)] = 0.0
		trot[np.where(trot<=0)] = 0.0
		fits.writeto('Trot_fit_double.fits', trot, header=hdrt, clobber=True)
		fits.writeto('Trot_error_double.fits', trot_error, header=hdrt, clobber=True)
		hdrt['bunit'] = 'km/s'
		fits.writeto('linew_low_double.fits', linew11, header=hdrt, clobber=True)
		fits.writeto('linew_upp_double.fits', linew22, header=hdrt, clobber=True)
		fits.writeto('peakv_double.fits', peakv, header=hdrt, clobber=True)
		#hdrt['bunit'] = '1'
		#fits.writeto('linew_ratio.fits', linewratio, header=hdrt, clobber=True)

def __avespec__(data1,data2,cutoff=0.009,ra_range=[0,1],dec_range=[0,1]):
	"""
	This function averages the spectra within a region and returns the averaged spectrum.
	"""
	tempdata1 = data1[:,dec_range[0]:dec_range[-1],ra_range[0]:ra_range[-1]]
	tempdata2 = data2[:,dec_range[0]:dec_range[-1],ra_range[0]:ra_range[-1]]
	#tempdata1[np.where(tempdata1<cutoff)] = 0.0
	#tempdata2[np.where(tempdata2<cutoff)] = 0.0

	wt_data1 = np.ones(tempdata1.shape[1:])
	wt_data2 = np.ones(tempdata2.shape[1:])
	wt_data1[np.where(tempdata1.max(axis=0)<cutoff)] = 0.0
	wt_data2[np.where(tempdata2.max(axis=0)<cutoff)] = 0.0
	wt_data1 = [wt_data1] * naxisv
	wt_data2 = [wt_data2] * naxisv

	#naxisv = data1.shape[0]
	#avedata1 = plab.np.empty(naxisv)
	#avedata2 = plab.np.empty(naxisv)
	#for i in np.arange[0:naxisv]:
	avedata1 = np.average(tempdata1, axis=(1,2), weights=wt_data1)
	avedata2 = np.average(tempdata2, axis=(1,2), weights=wt_data2)

	return avedata1, avedata2

def fit_spec(spec1, spec2, vaxis, cutoff=0.009, varyv=2, interactive=True, mode='single'):
	"""
	fit_spec(spec1, spec2)
	Derive the temperature with two spectra, instead of a cube.
	"""
	if interactive:
		plt.ion()
		f = plt.figure(figsize=(14,8))
		ax = f.add_subplot(111)

	#spec1 = spec1[19:-19] * 1000.0
	#spec2 = spec2[19:-19] * 1000.0
	spec1 = spec1[15:-15] * 1000.0
	spec2 = spec2[15:-15] * 1000.0
	cutoff = cutoff * 1000.0
	spec1[-1] = 0
	spec2[0] = 0
	vaxis = vaxis[15:-15]
	spec = np.concatenate((spec1, spec2))
	#spec[np.where(spec<=cutoff)] = 0.0
	vaxis = np.concatenate((vaxis, vaxis+30.0))
	spec = spec[0:-2]
	vaxis = vaxis[0:-2]

	#if interactive:
		#plt.plot(vaxis, spec, 'k+', label='Original')
		#plt.legend()
		#plt.show()

	unsatisfied = True
	while unsatisfied:
		if interactive:
			f.clear()
			plt.ion()
			plt.plot(vaxis, spec, 'k-', label='Spectrum')
			cutoff_line = [cutoff] * len(vaxis)
			cutoff_line_minus = [-1.0*cutoff] * len(vaxis)
			plt.plot(vaxis, cutoff_line, 'r-')
			plt.plot(vaxis, cutoff_line_minus, 'r-')
			#plt.plot(vaxis, np.zeros(len(vaxis)), 'b-')
			plt.xlabel(r'$V_\mathrm{lsr}$ (km s$^{-1}$)', fontsize=20, labelpad=20)
			plt.ylabel(r'$I_{\nu}$ (mJy beam$^{-1}$)', fontsize=20)
			plt.ylim([-5,25])
			#clickvalue = []
			if mode == 'single':
				cid = f.canvas.mpl_connect('button_press_event', onclick)
				raw_input('Click on the plot to select Vlsr...')
				print clickvalue
				if len(clickvalue) >= 1:
					print 'Please select at least one velocity! The last one will be used.'
					vlsr1 = clickvalue[-1]
				elif len(clickvalue) == 0:
					vlsr1 = 0.0
				print 'Or input one velocity manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 1:
					vlsr1 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The Vlsr is %0.2f' % vlsr1
				raw_input('Press any key to start fitting...')
				f.canvas.mpl_disconnect(cid)
				vlsr2 = 0.0
			elif mode == 'double':
				cid = f.canvas.mpl_connect('button_press_event', onclick)
				raw_input('Click on the plot to select Vlsrs...')
				print clickvalue
				if len(clickvalue) >= 2:
					print 'Please select at least two velocities! The last two will be used.'
					vlsr1,vlsr2 = clickvalue[-2],clickvalue[-1]
				elif len(clickvalue) == 1:
					vlsr1 = clickvalue[-1]
					vlsr2 = 0.0
				elif len(clickvalue) == 0:
					vlsr1,vlsr2 = 0.0,0.0
				print 'Or input two velocities manually:'
				manualv = raw_input()
				manualv = manualv.split()
				if len(manualv) == 2:
					vlsr1,vlsr2 = np.float_(manualv)
				else:
					print 'Invalid input...'
				print 'The two Vlsrs are %0.2f km/s and %0.2f km/s.' % (vlsr1,vlsr2)
				raw_input('Press any key to start fitting...')
				f.canvas.mpl_disconnect(cid)
			else:
				vlsr1,vlsr2 = 0.0,0.0
		else:
			if mode == 'single':
				if spec_low.max() >= cutoff:
					vlsr1 = __xcorrelate__(spec_low, vaxis_low)
					if vlsr1 <=82 or vlsr1 >=92:
						vlsr1 = 0.0
					if spec_low[np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
						vlsr1 = 0.0
					if spec_low[np.abs(vaxis_low - vlsr1 + 7.47385).argmin()] <= cutoff and spec_low[np.abs(vaxis_low - vlsr1 - 7.56923).argmin()] <= cutoff:
						vlsr1 = 0.0
					if spec_upp[np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
						vlsr1 = 0.0
				else:
					vlsr1 = 0.0
				vlsr2 = 0.0
			elif mode == 'double':
				vlsr1,vlsr2 = 86.0,88.0
			else:
				vlsr1,vlsr2 = 0.0,0.0


		# 18 parameters, but only 7 are indenpendent 
		params = Parameters()
		if vlsr1 != 0:
			params.add('peaki', value=30, min=0, max=50)
			#params.add('peaki', value=20, vary=False)
			params.add('tau11', value=1.0, min=0, max=20.0)
			if varyv > 0:
				params.add('peakv', value=vlsr1, min=vlsr1-varyv*onevpix, max=vlsr1+varyv*onevpix)
			elif varyv == 0:
				params.add('peakv', value=vlsr1, vary=False)
			params.add('sigmav', value=1.0, min=0, max=2.0)
			params.add('peaki_s1', expr="peaki*(1-exp(-tau11_s1))/(1-exp(-tau11))")
			params.add('tau11_s1', expr='tau11*0.278')
			params.add('peakv_s1', expr='peakv-7.47385')
			params.add('sigmav_s1', expr='sigmav')
			params.add('peaki_s2', expr='peaki_s1')
			params.add('tau11_s2', expr='tau11_s1')
			params.add('peakv_s2', expr='peakv+7.56923')
			params.add('sigmav_s2', expr='sigmav')
			params.add('peaki_upp', expr='peaki*(1-exp(-tau22))/(1-exp(-tau11))', min=0, max=50)
			#params.add('tau22', value=1.0, min=0, expr='<=tau11/0.282')
			params.add('tau22', value=1.0, min=0, max=10.0)
			params.add('peakv_upp', expr='peakv+30.0')
			#params.add('sigmav_upp', value=1.0, min=0, max=5.0)
			params.add('sigmav_upp', expr='sigmav')
			params.add('Trot', value=0.0, expr='-41.5/log(0.282*tau22/tau11)', min=0)
			# Add one more parameter: the total NH3 column density (column density of NH3 (1,1) * the partition function)
			params.add('Ntot', value=1e15, expr='6.6e14*Trot/23.6944955*tau11*2.355*sigmav*0.0138*exp(23.1/Trot)*Trot**1.5')
		# another 18 parameters for the second component
		if vlsr2 != 0:
			params.add('peaki_c2', value=30,  min=0, max=50)
			#params.add('peaki_c2', value=25, vary=False)
			params.add('tau11_c2', value=1.0, min=0, max=20.0)
			if varyv > 0:
				params.add('peakv_c2', value=vlsr2, min=vlsr2-varyv*onevpix, max=vlsr2+varyv*onevpix)
			elif varyv == 0:
				params.add('peakv_c2', value=vlsr2, vary=False)
			params.add('sigmav_c2', value=1.0, min=0, max=2.0)
			params.add('peaki_s1_c2', expr="peaki_c2*(1-exp(-tau11_s1_c2))/(1-exp(-tau11_c2))")
			params.add('tau11_s1_c2', expr='tau11_c2*0.278')
			params.add('peakv_s1_c2', expr='peakv_c2-7.47385')
			params.add('sigmav_s1_c2', expr='sigmav_c2')
			params.add('peaki_s2_c2', expr='peaki_s1_c2')
			params.add('tau11_s2_c2', expr='tau11_s1_c2')
			params.add('peakv_s2_c2', expr='peakv_c2+7.56923')
			params.add('sigmav_s2_c2', expr='sigmav_c2')
			params.add('peaki_upp_c2', expr='peaki_c2*(1-exp(-tau22_c2))/(1-exp(-tau11_c2))', min=0, max=50)
			#params.add('tau22_c2', value=1.0, min=0, expr='<=tau11_c2/0.282')
			params.add('tau22_c2', value=1.0, min=0, max=10.0)
			params.add('peakv_upp_c2', expr='peakv_c2+30.0')
			#params.add('sigmav_upp_c2', value=1.0, min=0, max=5.0)
			params.add('sigmav_upp_c2', expr="sigmav_c2")
			params.add('Trot_c2', value=0.0, expr='-41.5/log(0.282*tau22_c2/tau11_c2)', min=0)
			# Add one more parameter: the total NH3 column density (column density of NH3 (1,1) * the partition function)
			params.add('Ntot_c2', value=1e15, expr='6.6e14*Trot_c2/23.6944955*tau11_c2*2.355*sigmav_c2*0.0138*exp(23.1/Trot_c2)*Trot_c2**1.5')

		# do fit, here with leastsq model
		#spec[np.where(spec<=cutoff)] = 0.0
		if vlsr1 != 0 and vlsr2 != 0:
			result = minimize(__model_11_2c__, params, args=(vaxis, spec))
		elif vlsr1 != 0 or vlsr2 != 0:
			result = minimize(__model_11__, params, args=(vaxis, spec))
		else:
			unsatisfied = False
			continue
		
		final = spec + result.residual
		report_fit(params)

		if interactive:
			plt.plot(vaxis, final, 'r', label='Best-fitted model')
			if vlsr1 != 0 and vlsr2 != 0:
				final_c1 = __model_11__(params, vaxis, spec) + spec
				# Reconstruct the Guassian of 2nd component, using the fitting results.
				peaki_c2 = params['peaki_c2'].value
				tau11_c2 = params['tau11_c2'].value
				peakv_c2 = params['peakv_c2'].value
				sigmav_c2 = params['sigmav_c2'].value
				peaki_s1_c2 = params['peaki_s1_c2'].value
				tau11_s1_c2 = params['tau11_s1_c2'].value
				peakv_s1_c2 = params['peakv_s1_c2'].value
				sigmav_s1_c2 = params['sigmav_s1_c2'].value
				peaki_s2_c2 = params['peaki_s2_c2'].value
				tau11_s2_c2 = params['tau11_s2_c2'].value
				peakv_s2_c2 = params['peakv_s2_c2'].value
				sigmav_s2_c2 = params['sigmav_s2_c2'].value
				peaki_upp_c2 = params['peaki_upp_c2'].value
				tau22_c2 = params['tau22_c2'].value
				peakv_upp_c2 = params['peakv_upp_c2'].value
				sigmav_upp_c2 = params['sigmav_upp_c2'].value
				final_c2 = __gauss_tau__(vaxis,[peaki_c2,tau11_c2,peakv_c2,sigmav_c2]) + \
						__gauss_tau__(vaxis,[peaki_s1_c2,tau11_s1_c2,peakv_s1_c2,sigmav_s1_c2]) + \
						__gauss_tau__(vaxis,[peaki_s2_c2,tau11_s2_c2,peakv_s2_c2,sigmav_s2_c2]) + \
						__gauss_tau__(vaxis,[peaki_upp_c2,tau22_c2,peakv_upp_c2,sigmav_upp_c2])
				plt.plot(vaxis, final_c1, 'm--', label='1st component', linewidth=2)
				plt.plot(vaxis, final_c2, 'c--', label='2nd component', linewidth=2)
				plt.text(0.02, 0.9, r'1st $T_\mathrm{rot}$=%.1f$\pm$%.1f K' % (params['Trot'].value,params['Trot'].stderr), transform=ax.transAxes, color='m', fontsize=15)
				plt.text(0.02, 0.8, r'2nd $T_\mathrm{rot}$=%.1f$\pm$%.1f K' % (params['Trot_c2'].value,params['Trot_c2'].stderr), transform=ax.transAxes, color='c', fontsize=15)
			elif vlsr1 != 0 or vlsr2 != 0:
				plt.text(0.02, 0.9, r'$T_\mathrm{rot}$=%.1f$\pm$%.1f K' % (params['Trot'].value,params['Trot'].stderr), transform=ax.transAxes, color='r', fontsize=15)
			plt.legend()
			plt.show()
			print 'Is the fitting ok? y/n'
			yn = raw_input()
			if yn == 'y':
				unsatisfied = False 
			else:
				unsatisfied = True
			#raw_input('Press any key to continue...')
			f.clear()
		else:
			unsatisfied = False

def __readascii__(infile):
	temp = open(infile, 'r')
	text = temp.readlines()
	spec = np.empty(len(text))
	vaxis = np.empty(len(text))
	for line in range(len(text)):
		vaxis[line] = np.float(text[line].split()[0])
		spec[line] = np.float(text[line].split()[1])
	temp.close()
	del temp

	return spec, vaxis

data1,hdr1 = __nh3_read__('NH3_11_comb.cm.fits')
data2,hdr2 = __nh3_read__('NH3_22_comb.cm.regrid.fits')

naxisx = hdr1['naxis1']
naxisy = hdr1['naxis2']
naxisv = hdr1['naxis3']
bmaj_low = hdr1['bmaj']
bmin_low = hdr1['bmin']
bpa_low = hdr1['bpa']
#bmaj_upp = hdr2['bmaj']
#bmin_upp = hdr2['bmin']
#bpa_upp = hdr2['bpa']
#
nh3_info = __nh3_init__()

xaxis_low, yaxis_low, vaxis_low = __nh3_load_axes__(hdr1)
xaxis_upp, yaxis_upp, vaxis_upp = __nh3_load_axes__(hdr2)
vaxis_low = vaxis_low[::-1]

onexpix = hdr1['cdelt1']		# In unit of degree
oneypix = hdr1['cdelt2']		# In unit of degree
onevpix = hdr1['cdelt3']*0.001	# In unit of km/s

#ave1,ave2 = __avespec__(data1,data2,ra_range=np.arange(233,242),dec_range=np.arange(336,340))
#ave1,ave2 = __avespec__(data1,data2,ra_range=np.arange(239,247),dec_range=np.arange(319,328))

##############################################################
# mean temperature of mm5-p1:
#infile = csv.reader(open('./smacores/spectra/mm5_p1_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm5_p1_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm5_p1_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm5_p1_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm5_p1_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm5_p1_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.002, mode='single', varyv=0, interactive=True)

# mean temperature of mm5-p2:
#infile = csv.reader(open('./smacores/spectra/mm5_p2_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm5_p2_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm5_p2_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm5_p2_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm5_p2_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm5_p2_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.002, mode='single', varyv=0, interactive=True)

# mean temperature of mm5-p3:
#infile = csv.reader(open('./smacores/spectra/mm5_p3_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm5_p3_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm5_p3_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm5_p3_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm5_p3_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm5_p3_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.003, mode='single', varyv=0, interactive=True)

# mean temperature of mm1-p1:
#infile = csv.reader(open('./smacores/spectra/mm1_p1_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p1_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm1_p1_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p1_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm1_p1_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm1_p1_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='double', varyv=0)

# mean temperature of mm1-p2:
#infile = csv.reader(open('./smacores/spectra/mm1_p2_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p2_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm1_p2_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p2_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm1_p2_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm1_p2_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='double', varyv=0, interactive=True)

# mean temperature of mm1-p3:
#infile = csv.reader(open('./smacores/spectra/mm1_p3_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p3_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm1_p3_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p3_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm1_p3_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm1_p3_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='double', varyv=0, interactive=True)

# mean temperature of mm1-p4:
#infile = csv.reader(open('./smacores/spectra/mm1_p4_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p4_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm1_p4_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p4_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm1_p4_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm1_p4_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='double', varyv=0, interactive=True)

# mean temperature of mm1-p5:
#infile = csv.reader(open('./smacores/spectra/mm1_p5_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p5_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm1_p5_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p5_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm1_p5_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm1_p5_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='double', varyv=0, interactive=True)

# mean temperature of mm1-p6:
#infile = csv.reader(open('./smacores/spectra/mm1_p6_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p6_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm1_p6_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1_p6_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm1_p6_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm1_p6_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='double', varyv=0, interactive=True)

# mean temperature of mm7-p1:
#infile = csv.reader(open('./smacores/spectra/mm7_p1_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm7_p1_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm7_p1_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm7_p1_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm7_p1_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm7_p1_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='single', varyv=0, interactive=True)
#
## mean temperature of mm8-p1:
#infile = csv.reader(open('./smacores/spectra/mm8_p1_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm8_p1_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm8_p1_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm8_p1_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm8_p1_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm8_p1_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='single', varyv=0, interactive=True)
#
## mean temperature of mm8-p2:
#infile = csv.reader(open('./smacores/spectra/mm8_p2_11.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm8_p2_11_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm8_p2_22.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm8_p2_22_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#ave1, vaxis_11 = __readascii__('./smacores/spectra/mm8_p2_11_sorted.dat')
#ave2, vaxis_22 = __readascii__('./smacores/spectra/mm8_p2_22_sorted.dat')
#fit_spec(ave1, ave2, vaxis_low, cutoff=0.0025, mode='single', varyv=0, interactive=True)

# mean temperature of mm1-p1, vlaonly:
#infile = csv.reader(open('./smacores/spectra/mm1p1_11_vla.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1p1_11_vla_sorted.dat','w') 
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
#infile = csv.reader(open('./smacores/spectra/mm1p1_22_vla.dat'),delimiter=' ')
#sort = sorted(infile, key=lambda row: float(row[0]))
#file1 = open('./smacores/spectra/mm1p1_22_vla_sorted.dat','w')
#outfile = csv.writer(file1,delimiter=' ')
#outfile.writerows(sort)
#file1.close()
ave1, vaxis_11 = __readascii__('./smacores/spectra/mm1p1_11_vla_sorted.dat')
ave2, vaxis_22 = __readascii__('./smacores/spectra/mm1p1_22_vla_sorted.dat')
vaxis_11 = vaxis_11*0.001
fit_spec(ave1, ave2, vaxis_11, cutoff=0.0025, mode='double', varyv=0)

elapsed = (time.clock() - start)
print 'Stop the timer...'
print 'Time used: %0.0f seconds, or %0.1f minutes.' % (elapsed, elapsed/60.)

