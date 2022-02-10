# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from astropy import units as u

import healpy as hp

from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import qarray as qa

from .. import ops as ops

from ..observation import default_values as defaults

from ..pixels_io import write_healpix_fits

from ._helpers import (
    create_outdir,
    create_healpix_ring_satellite,
    create_satellite_data,
    create_fake_sky_alm,
    create_fake_beam_alm,
)



def create_fake_beam_alm(
    lmax=128,
    mmax=10,
    fwhm_x=10 * u.degree,
    fwhm_y=10 * u.degree,
    pol=True,
    separate_IQU=False,
    detB_beam=False 
):

    # pick an nside >= lmax to be sure that the a_lm will be fairly accurate
    nside = 2
    while nside < lmax:

        nside *= 2
    npix = 12 * nside ** 2
    pix = np.arange(npix)
    x,y,z  = hp.pix2vec(nside, pix, nest=False)  
    sigma_z = fwhm_x.to_value(u.radian) / np.sqrt(8 * np.log(2))
    sigma_y = fwhm_y.to_value(u.radian) / np.sqrt(8 * np.log(2))
    beam = np.exp(-((z ** 2 /2/sigma_z**2 + y ** 2 /2/ sigma_y**2)))
    beam[x < 0] = 0
    tmp_beam_map = np.zeros([3, npix ])
    tmp_beam_map[0] = beam
    tmp_beam_map[1] = beam
     
    
        
    bl, blm = hp.anafast(tmp_beam_map, lmax=lmax, iter=0, alm=True,pol=True)
    hp.rotate_alm(blm, psi=0, theta=-np.pi/2, phi=0)
    if   detB_beam : 
        # we make sure that the two detectors within the same pair encode two beams with the  flipped sign in Q   U beams  
        beam_map = hp.alm2map(blm, nside=nside, lmax=lmax , mmax=mmax)
        beam_map[1:] *= (-1) 
        blm=hp.map2alm (beam_map, lmax=lmax , mmax=mmax)
    if   separate_IQU:

        empty = np.zeros_like(tmp_beam_map[0])
        
        beam_map_I = np.vstack([tmp_beam_map[0], empty, empty])
        if   detB_beam : 

            beam_map_Q = np.vstack([empty, -tmp_beam_map[0], empty])
            beam_map_U = np.vstack([empty, empty, -tmp_beam_map[0]])
        else: 
            beam_map_Q = np.vstack([empty, tmp_beam_map[0], empty])
            beam_map_U = np.vstack([empty, empty, tmp_beam_map[0]])

        try:
            almI = hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax, verbose=False)
            almQ = hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax, verbose=False)
            almU = hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax, verbose=False)
             
            
        except TypeError:
            # older healpy which does not have verbose keyword
            almI = hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax )
            almQ = hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax )
            almU = hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax )
             
        hp.rotate_alm(almI, psi=0, theta=-np.pi/2, phi=0)
        hp.rotate_alm(almQ, psi=0, theta=-np.pi/2, phi=0)
        hp.rotate_alm(almU, psi=0, theta=-np.pi/2, phi=0)
        a_lm= [ almI, almQ, almU     ]
        return a_lm 
    else:
        return blm

class SimConviqtTest(MPITestCase):
    def setUp(self):
        
        
        np.random.seed(777)
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        self.nside = 64
        self.lmax = 128
        self.fwhm_sky = 10 * u.degree
        self.fwhm_beam = 15 * u.degree
        self.mmax = self.lmax
        self.fname_sky = os.path.join(self.outdir, "sky_alm.fits")
        self.fname_beam = os.path.join(self.outdir, "beam_alm.fits")

        self.rank = 0
        if self.comm is not None:
            self.rank = self.comm.rank

        if self.rank == 0:
            # Synthetic sky and beam (a_lm expansions)
            self.slm = create_fake_sky_alm(self.lmax, self.fwhm_sky)
            #self.slm[1:] = 0  # No polarization
            hp.write_alm(self.fname_sky, self.slm, lmax=self.lmax, overwrite=True)

            self.blm = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
            )
            self.blm_bottom= create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                detB_beam=True 
            )
            
            hp.write_alm(
                self.fname_beam,
                self.blm,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom.fits"),
                self.blm_bottom,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            blm_I, blm_Q, blm_U = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                separate_IQU=True,
            )
            blm_Ibot, blm_Qbot, blm_Ubot = create_fake_beam_alm(
                self.lmax,
                self.mmax,
                fwhm_x=self.fwhm_beam,
                fwhm_y=self.fwhm_beam,
                separate_IQU=True, detB_beam=True 
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_I000.fits"),
                blm_I,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_0I00.fits"),
                blm_Q,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_00I0.fits"),
                blm_U,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom_I000.fits"),
                blm_Ibot,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom_0I00.fits"),
                blm_Qbot,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            hp.write_alm(
                self.fname_beam.replace(".fits", "_bottom_00I0.fits"),
                blm_Ubot,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )

            # we explicitly store 3 separate beams for the T, E and B sky alm. 
            blm_T = np.zeros_like(self.blm)
            blm_T[0] = self.blm[0].copy()
            hp.write_alm(
                self.fname_beam.replace(".fits", "_T.fits"),
                blm_T,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            # in order to evaluate  
            # Q + iU ~  Sum[(b^E + ib^B)(a^E + ia^B)] , this implies
            # beamE = [0, blmE, -blmB] 
            
            blm_E = np.zeros_like(self.blm)
            blm_E[1] = self.blm[1] 
            blm_E[2] = -self.blm[2]  
            hp.write_alm(
                self.fname_beam.replace(".fits", "_E.fits"),
                blm_E,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
            #beamB = [0, blmB, blmE] 
            blm_B = np.zeros_like(self.blm)
            blm_B[1] =  self.blm[2] 
            blm_B[2] =  self.blm[1]  
            hp.write_alm(
                self.fname_beam.replace(".fits", "_B.fits"),
                blm_B,
                lmax=self.lmax,
                mmax_in=self.mmax,
                overwrite=True,
            )
        if self.comm is not None:
            self.comm.barrier()

        return
    
    
    def make_beam_file_dict(self,data) : 
        
        
        fname2= self.fname_beam.replace('.fits', '_bottom.fits')
        
        self.beam_file_dict={} 
        for det in data.obs[0].local_detectors:
            if det[-1]=="A" : 
                self.beam_file_dict[det] = self.fname_beam 
            else: 
                self.beam_file_dict[det] = fname2
                
                
        return 
            
    def test_sim_conviqt(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        #        data = create_healpix_ring_satellite(self.comm, nside=self.nside)
        data = create_satellite_data(self.comm , obs_time=120*u.min, pixel_per_process=2 )
        self. make_beam_file_dict(data)
            
        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key = defaults.det_data
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict= self.beam_file_dict,
            dxx=False,
            det_data=key,
            pol=True ,
            normalize_beam=True  ,
            fwhm=self.fwhm_sky,
        )
         
        sim_conviqt.apply(data)

        # Bin a map to study

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle= None ,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner.apply(data)

        # Study the map on the root process

        toast_bin_path = os.path.join(self.outdir, "toast_bin.fits")
        write_healpix_fits(data[binner.binned], toast_bin_path, nest=pixels.nest)

        toast_hits_path = os.path.join(self.outdir, "toast_hits.fits")
        write_healpix_fits(data[cov_and_hits.hits], toast_hits_path, nest=pixels.nest)

        fail = False

        if self.rank == 0:
            hitsfile = os.path.join(self.outdir, "toast_hits.fits")
            
            hdata = hp.read_map(hitsfile)
            
            footprint = np.ma.masked_not_equal(hdata, 0. ).mask
            
            mapfile = os.path.join(self.outdir, "toast_bin.fits")
            mdata = hp.read_map(mapfile, field=range(3) )

            
            deconv = 1 / hp.gauss_beam(
                self.fwhm_sky.to_value(u.radian),
                lmax=self.lmax,
                pol=False ,
            )

            smoothed = hp.alm2map(
                [ hp.almxfl(self.slm[ii] , deconv)  for ii in range(3)] ,
                self.nside,
                lmax=self.lmax,
                fwhm=self.fwhm_beam.to_value(u.radian),
                verbose=False,
                pixwin=False,
            )
            smoothed[:, ~footprint] =0 
            cl_out = hp.anafast(mdata, lmax=self.lmax)
            cl_smoothed = hp.anafast(smoothed, lmax=self.lmax)
            
            np.testing.assert_almost_equal(
            cl_smoothed[0] ,
                cl_out[0], decimal=2
        )
             
        return
    
    def test_sim_weighted_conviqt(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        #        data = create_healpix_ring_satellite(self.comm, nside=self.nside)
        data = create_satellite_data(self.comm , obs_time=120*u.min, pixel_per_process=2 )
        self. make_beam_file_dict(data)
            
        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

        key1 =  "conviqt"
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict= self.beam_file_dict,
            dxx=False,
            det_data=key1,
            pol=True ,
            normalize_beam= True    , 
            fwhm=self.fwhm_sky,
        )
         
        sim_conviqt.apply(data)

        key2 =  "wconviqt"
        
        sim_wconviqt = ops.SimWeightedConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file_dict = self.beam_file_dict,
            dxx=False,
            det_data=key2,
            pol=True ,
            normalize_beam=False   , #TODO:understand why if we set it to True we get both binned maps with nans 
            fwhm=self.fwhm_sky,
        )
         
        sim_wconviqt.apply(data)
        # Bin a map to study

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle= None ,
            detector_pointing=detpointing,
        )
        weights.apply(data)

        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        binner1 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key1,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner1.apply(data)
        binner2 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=key2,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner2.apply(data)
        # Study the map on the root process

        toast_bin_path = os.path.join(self.outdir, "toast_bin.conviqt.fits")
        write_healpix_fits(data[binner1.binned], toast_bin_path, nest=pixels.nest)
        toast_bin_path = os.path.join(self.outdir, "toast_bin.wconviqt.fits")
        write_healpix_fits(data[binner2.binned], toast_bin_path, nest=pixels.nest)
        
        toast_hits_path = os.path.join(self.outdir, "toast_hits.fits")
        write_healpix_fits(data[cov_and_hits.hits], toast_hits_path, nest=pixels.nest)

        fail = False
        if self.rank == 0:
            mapfile = os.path.join(self.outdir, "toast_bin.conviqt.fits")
            mdata = hp.read_map(mapfile, field=range(3) )
            mapfile = os.path.join(self.outdir, "toast_bin.wconviqt.fits")
            mdataw = hp.read_map(mapfile, field=range(3) )

            
            cl_out = hp.anafast(mdata, lmax=self.lmax)
            cl_outw = hp.anafast(mdataw, lmax=self.lmax)
            
            np.testing.assert_almost_equal(
              cl_out[0],
                cl_outw[0], decimal=2
        )
             
        return
   
    
"""   
    def test_TEBconvolution(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return

        # Create a fake scan strategy that hits every pixel once.
        data = create_healpix_ring_satellite(self.comm, nside=self.nside)

        # Generate timestreams

        detpointing = ops.PointingDetectorSimple()

       

        conviqt_key = "conviqt_tod"
        sim_conviqt = ops.SimConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file=self.fname_beam ,
            dxx=False,
            det_data=conviqt_key,
            #normalize_beam=True,
            fwhm=self.fwhm_sky,
        )
        sim_conviqt.exec(data)
        
        teb_conviqt_key  = "tebconviqt_tod"
        sim_teb = ops.SimTEBConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file=self.fname_beam ,
            dxx=False,
            det_data=teb_conviqt_key,
            #normalize_beam=True,
            fwhm=self.fwhm_sky,
        )
        sim_teb.exec(data)
        # Bin both signals into maps

        pixels = ops.PixelsHealpix(
            nside=self.nside,
            nest=False,
            detector_pointing=detpointing,
        )
        pixels.apply(data)
        weights = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing,
        )
        weights.apply(data)

        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        cov_and_hits = ops.CovarianceAndHits(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            rcond_threshold=1.0e-6,
            sync_type="alltoallv",
        )
        cov_and_hits.apply(data)

        
        
        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=teb_conviqt_key,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner.apply(data)
        path_teb = os.path.join(self.outdir, "toast_bin.tebconviqt.fits")
        
        write_healpix_fits(data[binner.binned], path_teb , nest=False)
        
        binner2 = ops.BinMap(
            pixel_dist="pixel_dist",
            covariance=cov_and_hits.covariance,
            det_data=conviqt_key,
            det_flags=None,
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
            sync_type="alltoallv",
        )
        binner2.apply(data)
        path_conviqt = os.path.join(self.outdir, "toast_bin.conviqt.fits")
        write_healpix_fits(data[binner2.binned], path_conviqt, nest=False)
        
        print(data[binner2.binned].data.shape) 
        np.testing.assert_almost_equal(data[binner2.binned].data ,    data[binner.binned].data, decimal=6)
        
####################         
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        fail = False

        if rank == 0:
            import matplotlib.pyplot as plt

            sky = hp.alm2map(self.slm , self.nside, lmax=self.lmax, verbose=False)
            beam = hp.alm2map(
                self.blm  ,
                self.nside,
                lmax=self.lmax,
                mmax=self.mmax,
                verbose=False,
            )

            #map_teb = hp.read_map(path_teb , field=range(3) )
            map_conviqt = hp.read_map(path_conviqt   )

            # For some reason, matplotlib hangs with multiple tasks,
            # even if only one writes.
            if self.comm is None or self.comm.size == 1:
                for i, pol in enumerate( 'I'):
                    fig = plt.figure(figsize=[12, 8])
                    nrow, ncol = 2, 2
                    hp.mollview(sky[i], title="input sky", sub=[nrow, ncol, 1])
                    hp.mollview(beam[i], title="beam", sub=[nrow, ncol, 2], rot=[0, 90])
                    #amp = np.amax(map_conviqt[i])/4
                    hp.mollview(
                        map_teb[i],
                        min=-amp,
                        max=amp,
                        title="TEB conviqt",
                        sub=[nrow, ncol, 3],
                    ) 
                    hp.mollview(
                        map_conviqt,
                        #min=-amp,
                        #max=amp,
                        title="conviqt",
                        sub=[nrow, ncol, 4],
                    )
                    outfile = os.path.join(self.outdir, f"map_comparison{pol}.png")
                    fig.savefig(outfile)
            for obs in data.obs:
                for det in obs.local_detectors:
                    tod_teb = obs.detdata[teb_conviqt_key][det]
                    tod_conviqt = obs.detdata[conviqt_key][det]
                    if not np.allclose(
                        tod_teb,
                        tod_conviqt,
                        rtol=1e-3,
                        atol=1e-3,
                    ):
                        import matplotlib.pyplot as plt
                        import pdb

                        pdb.set_trace()
                        fail = True
                        break
                if fail:
                    break
        if data.comm.comm_world is not None:
            fail = data.comm.comm_world.bcast(fail, root=0)

        self.assertFalse(fail)

        return
    
    def test_sim_hwp(self):
        if not ops.conviqt.available():
            print("libconviqt not available, skipping tests")
            return
        # Create a fake scan strategy that hits every pixel once.
        data = create_satellite_data(self.comm)
        # make a simple pointing matrix
        detpointing = ops.PointingDetectorSimple()
        # Generate timestreams
        sim_conviqt = ops.SimWeightedConviqt(
            comm=self.comm,
            detector_pointing=detpointing,
            sky_file=self.fname_sky,
            beam_file=self.fname_beam,
            dxx=False,
            hwp_angle="hwp_angle",
        )
        sim_conviqt.exec(data)
        return
    """