import pylab, os, aplpy, time
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
	xarray = pylab.np.arange(0,naxisx,1)
	yarray = pylab.np.arange(0,naxisy,1)
	w = wcs.WCS(header)
	# X axis
	y_mid = pylab.np.round(naxisy/2) - 1.0
	y_mid = [y_mid] * naxisx
	ra_pix = pylab.np.vstack([xarray,y_mid,pylab.np.zeros(naxisx)])
	ra_pix = ra_pix.transpose()
	xaxis = w.wcs_pix2world(ra_pix,1)
	xaxis = xaxis[:,0]
	# Y axis
	x_mid = pylab.np.round(naxisx/2) - 1.0
	x_mid = [x_mid] * naxisy
	dec_pix = pylab.np.vstack([x_mid,yarray,pylab.np.zeros(naxisy)])
	dec_pix = dec_pix.transpose()
	yaxis = w.wcs_pix2world(dec_pix,1)
	yaxis = yaxis[:,1]
	# V axis
	onevpix = header['cdelt3']*0.001
	v0 = header['crval3']*0.001
	v0pix = int(header['crpix3'])
	vaxis = onevpix * (pylab.np.arange(naxisv)+1-v0pix) + v0
	return xaxis, yaxis, vaxis

def __interp_upp2low__(data_upp):
	"""
	__interp_upp2low__(data_upp,xaxis_upp,)
	Interpolate the cube of the upper level to that of the lower level.
	"""
	delta = pylab.np.radians(hdr2['crval2'])
	onexpix_prj = onexpix/pylab.np.cos(delta)
	x0_upp = xaxis_upp[0]
	y0_upp = yaxis_upp[0]
	v0_upp = vaxis_upp[0]

	xaxis = (xaxis_low - x0_upp)/onexpix_prj
	yaxis = (yaxis_low - y0_upp)/oneypix
	vaxis = (vaxis_low - v0_upp)/onevpix

	Y, V, X = pylab.np.meshgrid(yaxis,vaxis,xaxis)
	data_upp = map_coordinates(data_upp,(V,Y,X),order=1)
	return data_upp
	print 'Interpolate the cube of the upper level to the lower'

def __regrid_spec__(spec, axis, stretch=10.0):
	"""
	"""
	length = len(axis)
	oldidx = pylab.np.linspace(0,length-1,length)
	newidx = pylab.np.linspace(0,length-1,length*stretch)
	temp = splrep(oldidx, axis, k=1)
	newaxis = splev(newidx,temp)
	temp = splrep(oldidx, spec, k=1)
	newspec = splev(newidx,temp)
	return newspec, newaxis

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
	#oldidx = pylab.np.linspace(0,vlength-1,vlength)
	#newidx = pylab.np.linspace(0,vlength-1,vlength*stretch)
	#temp = splrep(oldidx, vaxis)
	#newvaxis = splev(newidx,temp)
	#temp = splrep(oldidx, spec)
	#newspec = splev(newidx,temp)
	newspec,newvaxis = __regrid_spec__(spec,vaxis)
	kernel = pylab.np.zeros(vlength*stretch)
	sigmav = 1.0
	for i in pylab.np.arange(len(voff_lines)):
		kernel += pylab.np.exp(-(newvaxis-newvaxis[vlength*stretch/2]-voff_lines[i])**2/(2*sigmav**2))*tau_wts[i]
	lags = pylab.np.correlate(newspec,kernel,'same')
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

	firstguess = pylab.np.zeros([4,hdr_gbt['NAXIS2'],hdr_gbt['NAXIS1']])

	# Open Zoey's result
	temp = open(infile)
	text = temp.readlines()
	for block in pylab.np.arange(len(text)/14):
		indices = text[block*14].split()
		xno = pylab.np.int(indices[0])
		yno = pylab.np.int(indices[1])
		firstguess[0,yno,xno] = xaxis[xno]
		firstguess[1,yno,xno] = yaxis[yno]
		vhit1 = pylab.np.round(pylab.np.float(text[block*14+1].split()[2]))
		vhit2 = pylab.np.round(pylab.np.float(text[block*14+6].split()[2]))
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
	u  = ((axis-p[2])/pylab.np.abs(p[3]))**2
	f = -1.0 * p[1] * pylab.np.exp(-0.5*u)
	f = -1.0 * p[0] * pylab.np.expm1(f)
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
	t = pylab.np.arange(5000) * (taumax - taumin)/5000. + taumin
	rat_in = (1-pylab.np.exp(-t))/(1-pylab.np.exp(-aconst_in*t))
	print 'Ratios between ', rat_in[4999], ' and ', rat_in[0], ' are valid.'

	if mainflux >= cutoff:
		tau = taumin
		if sateflux >= cutoff:
			diff = 9999.
			best = 0.
			ratio = mainflux/sateflux
			best = pylab.np.abs(ratio - rat_in).argmin()
			tau = t[best]
	return tau

clickvalue = []
def onclick(event):
	print 'The Vlsr you select: %f' % event.xdata
	clickvalue.append(event.xdata)
	#clickvalue = clickvalue + [event.xdata]

def fit_2comp(data_low, data_upp, vaxis_11, vaxis_22, ra_range=[0,1], dec_range=[0,1], cutoff=0.010, varyv=2, writefits=False, interactive=False, mode='single'):
	"""
	fits_2comp(data_low, data_upp)
	Use Monte Carlo approach to derive the temperature based on two transitions.
	Also locate two velocity components, if there are any, and derive temperatures respectively.
	The velocity components are pre-identified based on GBT data by Zoey.
	"""
	if mode == 'single':
		trot = pylab.np.zeros([naxisy,naxisx])
		trot_error = pylab.np.zeros([naxisy,naxisx])
		linew11 = pylab.np.zeros([naxisy,naxisx])
		#linew22 = pylab.np.zeros([naxisy,naxisx])
		#linewratio = pylab.np.zeros([naxisy,naxisx])
		peakv = pylab.np.zeros([naxisy,naxisx])
	elif mode == 'double':
		trot = pylab.np.zeros([2,naxisy,naxisx])
		trot_error = pylab.np.zeros([2,naxisy,naxisx])
		linew11 = pylab.np.zeros([2,naxisy,naxisx])
		#linew22 = pylab.np.zeros([2,naxisy,naxisx])
		#linewratio = pylab.np.zeros([2,naxisy,naxisx])
		peakv = pylab.np.zeros([2,naxisy,naxisx])

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
		pylab.ion()
		f = pylab.figure(figsize=(14,8))
		ax = f.add_subplot(111)

	for i in dec_range:
		for j in ra_range:
			#print 'Start to fit pixel (%d,%d)' % (j,i)
			spec_low = data_low[10:-10,i,j]
			spec_upp = data_upp[25:-20,i,j]

			vaxis_low = vaxis_11[10:-10]
			vaxis_upp = vaxis_22[25:-20]
			
			spec_low = spec_low[::-1]
			spec_upp = spec_upp[::-1]
			vaxis_low = vaxis_low[::-1]
			vaxis_upp = vaxis_upp[::-1]

			spec = pylab.np.concatenate((spec_low, spec_upp))
			vaxis = pylab.np.concatenate((vaxis_low, vaxis_upp+40.0))

			#if interactive:
				#pylab.plot(vaxis, spec, 'k+', label='Original')
				#pylab.legend()
				#pylab.show()

			unsatisfied = True
			while unsatisfied:
				if interactive:
					f.clear()
					pylab.plot(vaxis, spec, 'k-', label='Original')
					cutoff_line = [cutoff] * len(vaxis)
					cutoff_line_minus = [-1.0*cutoff] * len(vaxis)
					pylab.plot(vaxis, cutoff_line, 'r-')
					pylab.plot(vaxis, cutoff_line_minus, 'r-')
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
							vlsr1,vlsr2 = pylab.np.float_(manualv)
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
							# If vlsr is out of range:
							if vlsr1 <=82 or vlsr1 >=92:
								vlsr1 = 0.0
							# If the intensity at vlsr is smaller than cutoff
							if spec_low[pylab.np.abs(vaxis_low - vlsr1).argmin()] <= cutoff:
								vlsr1 = 0.0
							# If the intensity at both satellites is smaller than cutoff
							if spec_low[pylab.np.abs(vaxis_low - vlsr1 + 7.47385).argmin()] <= cutoff and spec_low[pylab.np.abs(vaxis_low - vlsr1 - 7.56923).argmin()] <= cutoff:
								vlsr1 = 0.0
							# If the intensity of (2,2) is smaller than cutoff
							if spec_upp[pylab.np.abs(vaxis_upp - vlsr1).argmin()] <= cutoff:
								vlsr1 = 0.0
						else:
							vlsr1 = 0.0
						vlsr2 = 0.0
					elif mode == 'double':
						vlsr1,vlsr2 = 85.4,87.4
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
					params.add('tau22', value=1.0, min=0, max=10.0)
					params.add('peakv_upp', expr='peakv+40.0')
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
					params.add('tau22_c2', value=1.0, min=0, max=10.0)
					params.add('peakv_upp_c2', expr='peakv_c2+40.0')
					params.add('sigmav_upp_c2', expr="sigmav_c2")
					params.add('Trot_c2', value=0.0, expr='-41.5/log(0.282*tau22_c2/tau11_c2)', min=0)

				# do fit, here with leastsq model
				if vlsr1 != 0 and vlsr2 != 0:
					try:
						result = minimize(__model_11_2c__, params, args=(vaxis, spec))
					except RuntimeError:
						print 'Pixel (%d, %d) fitting failed...' % (j,i)
						continue
				elif vlsr1 != 0 or vlsr2 != 0:
					try:
						result = minimize(__model_11__, params, args=(vaxis, spec))
					except RuntimeError:
						print 'Pixel (%d, %d) fitting failed...' % (j,i)
						continue
				else:
					unsatisfied = False
					continue
				#print params['Trot'].value

				if interactive:
					final = spec + result.residual
					report_fit(params)
					pylab.plot(vaxis, final, 'r', label='Fitting result')
					if vlsr1 != 0 and vlsr2 != 0:
						final_c1 = __model_11__(params, vaxis, spec) + spec
						final_c2 = final - final_c1
						pylab.plot(vaxis, final_c1, 'm--', label='1st component', linewidth=2)
						pylab.plot(vaxis, final_c2, 'c--', label='2nd component', linewidth=2)
						pylab.text(0.05, 0.9, '1st Trot=%.1f K'%params['Trot'].value, transform=ax.transAxes, color='m')
						pylab.text(0.05, 0.8, '2nd Trot=%.1f K'%params['Trot_c2'].value, transform=ax.transAxes, color='c')
					elif vlsr1 != 0 or vlsr2 != 0:
						pylab.text(0.05, 0.9, 'Trot=%.1f K'%params['Trot'].value, transform=ax.transAxes, color='r')
					pylab.legend()
					pylab.show()
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

				if writefits:
					# write the temperature
					if mode == 'single':
						trot[i,j] = params['Trot'].value
						trot[pylab.np.where(trot>50.)] = 50.
						trot[pylab.np.where(trot<0.)] = 0.
						trot_error[i,j] = params['Trot'].stderr
						trot_error[pylab.np.where(trot_error>10.)] = 10.
						trot_error[pylab.np.where(trot_error<0.)] = 0.
						linew11[i,j] = params['sigmav'].value
						peakv[i,j] = params['peakv'].value
					if mode == 'double':
						trot[:,i,j] = [params['Trot'].value, params['Trot_c2'].value]
						trot[pylab.np.where(trot>50.)] = 50.
						trot[pylab.np.where(trot<0.)] = 0.
						trot_error[:,i,j] = [params['Trot'].stderr, params['Trot_c2'].value]
						trot_error[pylab.np.where(trot_error>10.)] = 10.
						trot_error[pylab.np.where(trot_error<0.)] = 0.
						linew11[:,i,j] = [params['sigmav'].value, params['sigmav_c2'].value]
						peakv[:,i,j] = [params['peakv'].value, params['peakv_c2'].value]

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
		fits.writeto('Trot_fit.fits', trot, header=hdrt, clobber=True)
		fits.writeto('Trot_error.fits', trot_error, header=hdrt, clobber=True)
		hdrt['bunit'] = 'km/s'
		fits.writeto('linew_low.fits', linew11, header=hdrt, clobber=True)
		fits.writeto('peakv.fits', peakv, header=hdrt, clobber=True)

data1,hdr1 = __nh3_read__('NH3_11_comb.cm.fits')
data2,hdr2 = __nh3_read__('NH3_22_comb.cm.regrid.fits')

naxisx = hdr1['naxis1']
naxisy = hdr1['naxis2']
naxisv = hdr1['naxis3']
bmaj_low = hdr1['bmaj']
bmin_low = hdr1['bmin']
bpa_low = hdr1['bpa']
bmaj_upp = hdr2['bmaj']
bmin_upp = hdr2['bmin']
bpa_upp = hdr2['bpa']

nh3_info = __nh3_init__()

xaxis_low, yaxis_low, vaxis_11 = __nh3_load_axes__(hdr1)
xaxis_upp, yaxis_upp, vaxis_22 = __nh3_load_axes__(hdr2)

onexpix = hdr1['cdelt1']		# In unit of degree
oneypix = hdr1['cdelt2']		# In unit of degree
onevpix = hdr1['cdelt3']*0.001	# In unit of km/s

#data2 = __interp_upp2low__(data2)
#fits.writeto('22.cm.regrid.fits',data2,hdr1,clobber=True)

#fit_2comp(data1,data2,ra_range=[153,154],dec_range=[148,149],cutoff=0.009,varyv=0.0,writefits=False,interactive=True)

#fit_2comp(data1,data2,vaxis_11,vaxis_22,ra_range=pylab.np.arange(236,240),dec_range=pylab.np.arange(337,339),cutoff=0.009,varyv=0.0,writefits=False,interactive=True,mode='double')

#fit_2comp(data1,data2,ra_range=pylab.np.arange(200,280),dec_range=pylab.np.arange(280,380),cutoff=0.009,varyv=0.5,writefits=True,interactive=False,mode='double')

fit_2comp(data1,data2,vaxis_11,vaxis_22,ra_range=pylab.np.arange(naxisx),dec_range=pylab.np.arange(naxisy),cutoff=0.009,varyv=0,writefits=True,interactive=False,mode='single')

elapsed = (time.clock() - start)
print 'Stop the timer...'
print 'Time used: %0.0f seconds, or %0.1f minutes.' % (elapsed, elapsed/60.)
print '\a'

#fig = aplpy.FITSFigure('Trot_fit.fits')
#x_c = hdr2['crval1']
#y_c = hdr2['crval2']
#print 'Center of the image is', x_c, y_c
#fig.recenter(x=x_c, y=y_c, radius=0.05)
#fig.show_colorscale(vmin=8,vmax=26)
#fig.add_colorbar()
#fig.save('trot.png')
#
#fig = aplpy.FITSFigure('Trot_error.fits')
#x_c = hdr2['crval1']
#y_c = hdr2['crval2']
#print 'Center of the image is', x_c, y_c
#fig.recenter(x=x_c, y=y_c, radius=0.05)
#fig.show_colorscale(vmin=0,vmax=1.5)
#fig.add_colorbar()
#fig.save('trot_error.png')
#
#fig = aplpy.FITSFigure('linew_low.fits')
#x_c = hdr2['crval1']
#y_c = hdr2['crval2']
#print 'Center of the image is', x_c, y_c
#fig.recenter(x=x_c, y=y_c, radius=0.05)
#fig.show_colorscale(vmin=0,vmax=2)
#fig.add_colorbar()
#fig.save('lw_low.png')
#
#fig = aplpy.FITSFigure('linew_upp.fits')
#x_c = hdr2['crval1']
#y_c = hdr2['crval2']
#print 'Center of the image is', x_c, y_c
#fig.recenter(x=x_c, y=y_c, radius=0.05)
#fig.show_colorscale(vmin=0,vmax=2)
#fig.add_colorbar()
#fig.save('lw_upp.png')
#
#raw_input('Press to any key to finish...')

