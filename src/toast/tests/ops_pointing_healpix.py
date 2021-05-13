# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from .mpi import MPITestCase

import os

import healpy as hp
import numpy as np

from .._libtoast import pointing_matrix_healpix

from .. import qarray as qa

from ..healpix import HealpixPixels

from .. import ops as ops

from ..intervals import Interval, IntervalList

from ._helpers import create_outdir, create_satellite_data


class PointingHealpixTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_pointing_matrix_healpix2(self):
        nside = 64
        npix = 12 * nside ** 2
        hpix = HealpixPixels(64)
        nest = True
        phivec = np.radians(
            [-360, -270, -180, -135, -90, -45, 0, 45, 90, 135, 180, 270, 360]
        )
        nsamp = phivec.size
        eps = 0.0
        cal = 1.0
        mode = "IQU"
        nnz = 3
        hwpang = np.zeros(nsamp)
        flags = np.zeros(nsamp, dtype=np.uint8)
        pixels = np.zeros(nsamp, dtype=np.int64)
        weights = np.zeros([nsamp, nnz], dtype=np.float64)
        theta = np.radians(135)
        psi = np.radians(135)
        quats = []
        xaxis, yaxis, zaxis = np.eye(3)
        for phi in phivec:
            phirot = qa.rotation(zaxis, phi)
            quats.append(qa.from_angles(theta, phi, psi))
        quats = np.vstack(quats)
        pointing_matrix_healpix(
            hpix,
            nest,
            eps,
            cal,
            mode,
            quats.reshape(-1),
            hwpang,
            flags,
            pixels,
            weights.reshape(-1),
        )
        failed = False
        bad = np.logical_or(pixels < 0, pixels > npix - 1)
        nbad = np.sum(bad)
        if nbad > 0:
            print(
                "{} pixels are outside of the map. phi = {} deg".format(
                    nbad, np.degrees(phivec[bad])
                )
            )
            failed = True
        self.assertFalse(failed)
        return

    def test_pointing_matrix_healpix(self):
        nside = 64
        hpix = HealpixPixels(64)
        nest = True
        psivec = np.radians([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        # psivec = np.radians([-180, 180])
        nsamp = psivec.size
        eps = 0.0
        cal = 1.0
        mode = "IQU"
        nnz = 3
        hwpang = np.zeros(nsamp)
        flags = np.zeros(nsamp, dtype=np.uint8)
        pixels = np.zeros(nsamp, dtype=np.int64)
        weights = np.zeros([nsamp, nnz], dtype=np.float64)
        pix = 49103
        theta, phi = hp.pix2ang(nside, pix, nest=nest)
        xaxis, yaxis, zaxis = np.eye(3)
        thetarot = qa.rotation(yaxis, theta)
        phirot = qa.rotation(zaxis, phi)
        pixrot = qa.mult(phirot, thetarot)
        quats = []
        for psi in psivec:
            psirot = qa.rotation(zaxis, psi)
            quats.append(qa.mult(pixrot, psirot))
        quats = np.vstack(quats)
        pointing_matrix_healpix(
            hpix,
            nest,
            eps,
            cal,
            mode,
            quats.reshape(-1),
            hwpang,
            flags,
            pixels,
            weights.reshape(-1),
        )
        weights_ref = []
        for quat in quats:
            theta, phi, psi = qa.to_angles(quat)
            weights_ref.append(np.array([1, np.cos(2 * psi), np.sin(2 * psi)]))
        weights_ref = np.vstack(weights_ref)
        failed = False
        for w1, w2, psi, quat in zip(weights_ref, weights, psivec, quats):
            # print("\npsi = {}, quat = {} : ".format(psi, quat), end="")
            if not np.allclose(w1, w2):
                print(
                    "Pointing weights do not agree: {} != {}".format(w1, w2), flush=True
                )
                failed = True
            else:
                # print("Pointing weights agree: {} == {}".format(w1, w2), flush=True)
                pass
        self.assertFalse(failed)
        return

    def test_pointing_matrix_healpix_hwp(self):
        nside = 64
        hpix = HealpixPixels(64)
        nest = True
        psivec = np.radians([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        nsamp = len(psivec)
        eps = 0.0
        cal = 1.0
        mode = "IQU"
        nnz = 3
        flags = np.zeros(nsamp, dtype=np.uint8)
        pix = 49103
        theta, phi = hp.pix2ang(nside, pix, nest=nest)
        xaxis, yaxis, zaxis = np.eye(3)
        thetarot = qa.rotation(yaxis, theta)
        phirot = qa.rotation(zaxis, phi)
        pixrot = qa.mult(phirot, thetarot)
        quats = []
        for psi in psivec:
            psirot = qa.rotation(zaxis, psi)
            quats.append(qa.mult(pixrot, psirot))
        quats = np.vstack(quats)

        # First with HWP angle == 0.0
        hwpang = np.zeros(nsamp)
        pixels_zero = np.zeros(nsamp, dtype=np.int64)
        weights_zero = np.zeros([nsamp, nnz], dtype=np.float64)
        pointing_matrix_healpix(
            hpix,
            nest,
            eps,
            cal,
            mode,
            quats.reshape(-1),
            hwpang,
            flags,
            pixels_zero,
            weights_zero.reshape(-1),
        )

        # Now passing hwpang == None
        pixels_none = np.zeros(nsamp, dtype=np.int64)
        weights_none = np.zeros([nsamp, nnz], dtype=np.float64)
        pointing_matrix_healpix(
            hpix,
            nest,
            eps,
            cal,
            mode,
            quats.reshape(-1),
            None,
            flags,
            pixels_none,
            weights_none.reshape(-1),
        )
        # print("")
        # for i in range(nsamp):
        #     print(
        #         "HWP zero:  {} {} | {} {} {}".format(
        #             psivec[i],
        #             pixels_zero[i],
        #             weights_zero[i][0],
        #             weights_zero[i][1],
        #             weights_zero[i][2],
        #         )
        #     )
        #     print(
        #         "    none:  {} {} | {} {} {}".format(
        #             psivec[i],
        #             pixels_none[i],
        #             weights_none[i][0],
        #             weights_none[i][1],
        #             weights_none[i][2],
        #         )
        #     )
        failed = False
        if not np.all(np.equal(pixels_zero, pixels_none)):
            print("HWP pixels do not agree {} != {}".format(pixels_zero, pixels_none))
            failed = True

        if not np.allclose(weights_zero, weights_none):
            print(
                "HWP weights do not agree {} != {}".format(weights_zero, weights_none)
            )
            failed = True

        self.assertFalse(failed)
        return

    def test_hpix_simple(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        detpointing = ops.PointingDetectorSimple()
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", hwp_angle="hwp_angle", detector_pointing=detpointing,
        )
        pointing.apply(data)

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_hpix_simple_info"), "w")
        data.info(handle=handle)
        if rank == 0:
            handle.close()

    def test_hpix_interval(self):
        data = create_satellite_data(self.comm)

        full_intervals = "full_intervals"
        half_intervals = "half_intervals"
        for obs in data.obs:
            times = obs.shared["times"]
            nsample = len(times)
            intervals1 = [
                Interval(start=times[0], stop=times[-1], first=0, last=nsample - 1,)
            ]
            intervals2 = [
                Interval(
                    start=times[0],
                    stop=times[nsample // 2],
                    first=0,
                    last=nsample // 2,
                )
            ]
            obs.intervals[full_intervals] = IntervalList(times, intervals=intervals1)
            obs.intervals[half_intervals] = IntervalList(times, intervals=intervals2)

        detpointing = ops.PointingDetectorSimple(view=half_intervals)
        pointing = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle="hwp_angle",
            detector_pointing=detpointing,
            view=full_intervals,
        )
        with self.assertRaises(RuntimeError):
            pointing.apply(data)

        detpointing = ops.PointingDetectorSimple(view=full_intervals)
        pointing = ops.PointingHealpix(
            nside=64,
            mode="IQU",
            hwp_angle="hwp_angle",
            detector_pointing=detpointing,
            view=half_intervals,
        )
        pointing.apply(data)

    def test_hpix_hwpnull(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        detpointing = ops.PointingDetectorSimple()
        pointing = ops.PointingHealpix(
            nside=64, mode="IQU", detector_pointing=detpointing
        )
        pointing.apply(data)

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        handle = None
        if rank == 0:
            handle = open(os.path.join(self.outdir, "out_test_hpix_hwpnull"), "w")
        data.info(handle=handle)
        if rank == 0:
            handle.close()
