# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import scipy.interpolate
import traitlets
from astropy import units as u
from numpy.core.fromnumeric import size

from .. import qarray as qa
from ..atm import available_atm, available_utils
from ..mpi import MPI
from ..observation import default_values as defaults
from ..observation_dist import global_interval_times
from ..timing import GlobalTimers, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, Timer
from .operator import Operator

if available_atm:
    from ..atm import AtmSim

if available_utils:
    from ..atm import atm_absorption_coefficient_vec, atm_atmospheric_loading_vec


@trait_docs
class GenerateAtmosphere(Operator):
    """Operator which simulates or loads atmosphere realizations

    For each observing session, this operator simulates (or loads from disk) the
    atmosphere realization.  The simulated data is stored in the Data container.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    boresight = Unicode(
        defaults.boresight_azel, help="Observation shared key for Az/El boresight"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    output = Unicode(
        "atm_sim", help="Data key to store the dictionary of sims per session"
    )

    polarization_fraction = Float(
        0,
        help="Polarization fraction (only Q polarization).",
    )

    turnaround_interval = Unicode("turnaround", help="Interval name for turnarounds")

    realization = Int(
        0, help="If simulating multiple realizations, the realization index"
    )

    component = Int(
        123456, help="The component index to use for this atmosphere simulation"
    )

    lmin_center = Quantity(
        0.01 * u.meter, help="Kolmogorov turbulence dissipation scale center"
    )

    lmin_sigma = Quantity(
        0.001 * u.meter, help="Kolmogorov turbulence dissipation scale sigma"
    )

    lmax_center = Quantity(
        10.0 * u.meter, help="Kolmogorov turbulence injection scale center"
    )

    lmax_sigma = Quantity(
        10.0 * u.meter, help="Kolmogorov turbulence injection scale sigma"
    )

    gain = Float(1e-5, help="Scaling applied to the simulated TOD")

    zatm = Quantity(40000.0 * u.meter, help="Atmosphere extent for temperature profile")

    zmax = Quantity(
        2000.0 * u.meter, help="Atmosphere extent for water vapor integration"
    )

    xstep = Quantity(100.0 * u.meter, help="Size of volume elements in X direction")

    ystep = Quantity(100.0 * u.meter, help="Size of volume elements in Y direction")

    zstep = Quantity(100.0 * u.meter, help="Size of volume elements in Z direction")

    z0_center = Quantity(
        2000.0 * u.meter, help="Central value of the water vapor distribution"
    )

    z0_sigma = Quantity(0.0 * u.meter, help="Sigma of the water vapor distribution")

    wind_dist = Quantity(
        3000.0 * u.meter,
        help="Maximum wind drift before discarding the volume and creating a new one",
    )

    fade_time = Quantity(
        60.0 * u.s,
        help="Fade in/out time to avoid a step at wind break.",
    )

    sample_rate = Quantity(
        None,
        allow_none=True,
        help="Rate at which to sample atmospheric TOD before interpolation.  "
        "Default is no interpolation.",
    )

    nelem_sim_max = Int(10000, help="Controls the size of the simulation slices")

    cache_dir = Unicode(
        None,
        allow_none=True,
        help="Directory to use for loading / saving atmosphere realizations",
    )

    overwrite_cache = Bool(
        False, help="If True, redo and overwrite any cached atmospheric realizations."
    )

    cache_only = Bool(
        False, help="If True, only cache the atmosphere, do not observe it."
    )

    debug_spectrum = Bool(False, help="If True, dump out Kolmogorov debug files")

    debug_snapshots = Bool(
        False, help="If True, dump snapshots of the atmosphere slabs to pickle files"
    )

    debug_plots = Bool(False, help="If True, make plots of the debug snapshots")

    add_loading = Bool(True, help="Add elevation-dependent loading.")

    field_of_view = Quantity(
        None,
        allow_none=True,
        help="Override the focalplane field of view",
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        # Since each process group has the same set of observations, we use the group
        # communicator for collectively simulating the atmosphere slab for each
        # observation.
        comm = data.comm.comm_group
        group = data.comm.group
        rank = data.comm.group_rank
        comm_node = data.comm.comm_group_node
        comm_node_rank = data.comm.comm_group_node_rank

        # Name of the intervals for ranges valid for a given wind chunk
        wind_intervals = "wind"

        # Name of a view that combines user input and wind breaks
        if self.view is None:
            temporary_view = wind_intervals
        else:
            temporary_view = "temporary_view"

        # The atmosphere sims are created and stored for each observing session.
        # This data key contains a dictionary of sims, keyed on session name.
        if self.output not in data:
            data[self.output] = dict()

        # Split the observations according to their observing session.  This split
        # also checks that every session has a name.
        data_sessions = data.split(obs_session_name=True, require_full=True)

        for sname, sdata in data_sessions.items():
            # Prefix for logging
            log_prefix = f"{group} : {sname} : "

            # List of simulated slabs for each wind interval
            data[self.output][sname] = list()

            # For each session, check that the observations have the same site.
            site = None
            for ob in sdata.obs:
                if site is None:
                    site = ob.telescope.site
                elif ob.telescope.site != site:
                    msg = f"Different sites found for observations within the same "
                    msg += f"session: {site} != {ob.telescope.site}"
                    log.error(msg)
                    raise RuntimeError(msg)

            if not hasattr(site, "weather") or site.weather is None:
                raise RuntimeError(
                    "Cannot simulate atmosphere for sites without weather"
                )
            weather = site.weather

            # The session is by definition the same for all observations in this split.
            # Use the first observation to compute some quantities common to all.

            first_obs = sdata.obs[0]

            # The full time range of this session.  The processes are arranged
            # in a grid, and processes along the same column have the same timestamps.
            # So the first process of the group has the earliest times and the last
            # process of the group has the latest times.

            times = first_obs.shared[self.times]

            tmin_tot = times[0]
            tmax_tot = times[-1]
            if comm is not None:
                tmin_tot = comm.bcast(tmin_tot, root=0)
                tmax_tot = comm.bcast(tmax_tot, root=(data.comm.group_size - 1))

            # RNG values
            key1, key2, counter1, counter2 = self._get_rng_keys(first_obs)

            # Path to the cache location
            cachedir = self._get_cache_dir(first_obs, comm)

            log.debug_rank(f"{log_prefix}Setting up atmosphere simulation", comm=comm)

            # Although each observation in the session has the same boresight pointing,
            # they have independent fields of view / extent.  We find the maximal
            # Az / El range across all observations in the session.

            azmin = None
            azmax = None
            elmin = None
            elmax = None
            for ob in sdata.obs:
                ob_azmin, ob_azmax, ob_elmin, ob_elmax = self._get_scan_range(
                    first_obs, comm
                )
                if azmin is None:
                    azmin = ob_azmin
                    azmax = ob_azmax
                    elmin = ob_elmin
                    elmax = ob_elmax
                else:
                    azmin = np.min(azmin, ob_azmin)
                    azmax = np.max(azmax, ob_azmax)
                    elmin = np.min(elmin, ob_elmin)
                    elmax = np.max(elmax, ob_elmax)

            # Loop over the time span in "wind_time"-sized chunks.
            # wind_time is intended to reflect the correlation length
            # in the atmospheric noise.

            wind_times = list()

            tmr = Timer()
            if comm is not None:
                comm.Barrier()
            tmr.start()

            tmin = tmin_tot
            istart = 0
            counter1start = counter1

            while tmin < tmax_tot:
                if comm is not None:
                    comm.Barrier()
                istart, istop, tmax = self._get_time_range(
                    tmin, istart, times, tmax_tot, first_obs, weather
                )
                wind_times.append((tmin, tmax))

                if rank == 0:
                    log.debug(
                        f"{log_prefix}Instantiating atmosphere for t = "
                        f"{tmin - tmin_tot:10.1f} s - {tmax - tmin_tot:10.1f} s "
                        f"out of {tmax_tot - tmin_tot:10.1f} s"
                    )

                ind = slice(istart, istop)
                nind = istop - istart

                rmin = 0
                rmax = 100
                scale = 10
                counter2start = counter2
                counter1 = counter1start
                xstep_current = u.Quantity(self.xstep)
                ystep_current = u.Quantity(self.ystep)
                zstep_current = u.Quantity(self.zstep)

                sim_list = list()

                while rmax < 100000:
                    sim, counter2 = self._simulate_atmosphere(
                        weather,
                        scan_range,
                        tmin,
                        tmax,
                        comm,
                        comm_node,
                        comm_node_rank,
                        key1,
                        key2,
                        counter1,
                        counter2start,
                        cachedir,
                        log_prefix,
                        tmin_tot,
                        tmax_tot,
                        xstep_current,
                        ystep_current,
                        zstep_current,
                        rmin,
                        rmax,
                    )
                    sim_list.append(sim)

                    if self.debug_plots or self.debug_snapshots:
                        self._plot_snapshots(
                            sim,
                            log_prefix,
                            ob.name,
                            scan_range,
                            tmin,
                            tmax,
                            comm,
                            rmin,
                            rmax,
                        )

                    rmin = rmax
                    rmax *= scale
                    xstep_current *= np.sqrt(scale)
                    ystep_current *= np.sqrt(scale)
                    zstep_current *= np.sqrt(scale)
                    counter1 += 1

                data[self.output][sname].append(sim_list)

                if self.debug_tod:
                    self._save_tod(ob, times, istart, nind, ind, comm)
                tmin = tmax

            # Create the wind intervals
            ob.intervals.create_col(wind_intervals, wind_times, ob.shared[self.times])

            # Create temporary intervals by combining views
            if temporary_view != wind_intervals:
                ob.intervals[temporary_view] = (
                    ob.intervals[view] & ob.intervals[wind_intervals]
                )

            if temporary_view != wind_intervals:
                del ob.intervals[temporary_view]
            del ob.intervals[wind_intervals]

    def _get_rng_keys(self, obs):
        """Get random number keys and counters for an observing session.

        The session and telescope UID values are 32bit integers.  The realization
        index is typically smaller, and should fit within 2^16.  The component index
        is a small integer.

        The random number generator accepts a key and a counter, each made of two 64bit
        integers.  For a given observation we set these as:
            key 1 = site UID * 2^32 + telescope UID
            key 2 = session UID * 2^32 + realization * 2^16 + component
            counter 1 = hierarchical cone counter
            counter 2 = 0 (this is incremented per RNG stream sample)

        Args:
            obs (Observation):  One observation in the session.

        Returns:
            (tuple): The key1, key2, counter1, counter2 to use.

        """
        telescope = obs.telescope.uid
        site = obs.telescope.site.uid
        session = obs.session.uid

        # site UID in higher bits, telescope UID in lower bits
        key1 = site * 2**32 + telescope

        # Observation UID in higher bits, realization and component in lower bits
        key2 = session * 2**32 + self.realization * 2**16 + self.component

        # This tracks the number of cones simulated due to the wind speed.
        counter1 = 0

        # Starting point for the observation, incremented for each slice.
        counter2 = 0

        return key1, key2, counter1, counter2

    def _get_cache_dir(self, obs, comm):
        session_id = obs.session.uid
        if self.cache_dir is None:
            cachedir = None
        else:
            # The number of atmospheric realizations can be large.  Use
            # sub-directories under cachedir.
            subdir = str(int((session_id % 1000) // 100))
            subsubdir = str(int((session_id % 100) // 10))
            subsubsubdir = str(session_id % 10)
            cachedir = os.path.join(self.cache_dir, subdir, subsubdir, subsubsubdir)
            if (comm is None) or (comm.rank == 0):
                # Handle a rare race condition when two process groups
                # are creating the cache directories at the same time
                while True:
                    try:
                        os.makedirs(cachedir, exist_ok=True)
                    except OSError:
                        continue
                    except FileNotFoundError:
                        continue
                    else:
                        break
        return cachedir

    @function_timer
    def _get_scan_range(self, obs):
        if self.field_of_view is not None:
            fov = self.field_of_view
        else:
            fov = obs.telescope.focalplane.field_of_view
        fp_radius = 0.5 * fov.to_value(u.radian)

        # Work in parallel across each process column, which have the same
        # slice of time/samples

        if obs.comm_col is None:
            rank = 0
            ntask = 1
        else:
            rank = obs.comm_col_rank
            ntask = obs.comm_col_size

        # Create a fake focalplane of detectors in a circle around the boresight

        xaxis, yaxis, zaxis = np.eye(3)
        ndet = 64
        phidet = np.linspace(0, 2 * np.pi, ndet, endpoint=False)
        detquats = []
        thetarot = qa.rotation(yaxis, fp_radius)
        for phi in phidet:
            phirot = qa.rotation(zaxis, phi)
            detquat = qa.mult(phirot, thetarot)
            detquats.append(detquat)

        # Get fake detector pointing

        az = []
        el = []
        quats = obs.shared[self.boresight][rank::ntask].copy()
        for detquat in detquats:
            vecs = qa.rotate(qa.mult(quats, detquat), zaxis)
            theta, phi = hp.vec2ang(vecs)
            az.append(2 * np.pi - phi)
            el.append(np.pi / 2 - theta)
        az = np.unwrap(np.hstack(az))
        el = np.hstack(el)

        # find the extremes

        azmin = np.amin(az)
        azmax = np.amax(az)
        elmin = np.amin(el)
        elmax = np.amax(el)

        if azmin < -2 * np.pi:
            azmin += 2 * np.pi
            azmax += 2 * np.pi
        elif azmax > 2 * np.pi:
            azmin -= 2 * np.pi
            azmax -= 2 * np.pi

        # Combine results across all processes in the group

        if obs.comm.comm_group is not None:
            azmin = obs.comm.comm_group.allreduce(azmin, op=MPI.MIN)
            azmax = obs.comm.comm_group.allreduce(azmax, op=MPI.MAX)
            elmin = obs.comm.comm_group.allreduce(elmin, op=MPI.MIN)
            elmax = obs.comm.comm_group.allreduce(elmax, op=MPI.MAX)

        return azmin * u.radian, azmax * u.radian, elmin * u.radian, elmax * u.radian

    @function_timer
    def _get_time_range(self, tmin, istart, times, tmax_tot, obs, weather):
        # Do this calculation on one process.  Get the times and intervals here.

        turn_ilist = global_interval_times(
            obs.dist, obs.intervals, self.turnaround_interval, join=False
        )

        all_times = times
        if obs.comm_row_size > 1 and obs.comm_col_rank == 0:
            all_times = obs.comm_row.gather(times, root=0)
            if obs.comm_row_rank == 0:
                all_times = np.concatenate(all_times)

        # FIXME:  The code below is explicitly looping over numpy arrays.  If this is
        # too slow, we should move to using numpy functions like searchsorted, etc.

        if obs.comm.group_rank == 0:
            while all_times[istart] < tmin:
                istart += 1

            # Translate the wind speed to time span of a correlated interval
            wx = weather.west_wind.to_value(u.meter / u.second)
            wy = weather.south_wind.to_value(u.meter / u.second)
            w = np.sqrt(wx**2 + wy**2)
            wind_time = self.wind_dist.to_value(u.meter) / w

            tmax = tmin + wind_time
            if tmax < tmax_tot:
                # Extend the scan to the next turnaround
                istop = istart
                while istop < len(all_times) and all_times[istop] < tmax:
                    istop += 1
                iturn = 0
                while iturn < len(turn_ilist) - 1 and (
                    all_times[istop] > turn_ilist[iturn].stop
                ):
                    iturn += 1
                if all_times[istop] > turn_ilist[iturn].stop:
                    # We are past the last turnaround.
                    # Extend to the end of the observation.
                    istop = len(all_times)
                    tmax = tmax_tot
                else:
                    # Stop time is either before or in the middle of the turnaround.
                    # Extend to the start of the turnaround.
                    while istop < len(all_times) and (
                        all_times[istop] < turn_ilist[iturn].start
                    ):
                        istop += 1
                    if istop < len(all_times):
                        tmax = all_times[istop]
                    else:
                        tmax = tmax_tot
            else:
                tmax = tmax_tot
                istop = len(all_times)
            tmax = np.ceil(tmax)

        if obs.comm.comm_group is not None:
            istart = obs.comm.comm_group.bcast(istart, root=0)
            istop = obs.comm.comm_group.bcast(istop, root=0)
            tmax = obs.comm.comm_group.bcast(tmax, root=0)

        return istart, istop, tmax

    @function_timer
    def _simulate_atmosphere(
        self,
        weather,
        scan_range,
        tmin,
        tmax,
        comm,
        comm_node,
        comm_node_rank,
        key1,
        key2,
        counter1,
        counter2,
        cachedir,
        prefix,
        tmin_tot,
        tmax_tot,
        xstep,
        ystep,
        zstep,
        rmin,
        rmax,
    ):
        log = Logger.get()
        rank = 0
        tmr = Timer()
        if comm is not None:
            rank = comm.rank
            comm.Barrier()
        tmr.start()

        T0_center = weather.air_temperature
        wx = weather.west_wind
        wy = weather.south_wind
        w_center = np.sqrt(wx**2 + wy**2)
        wdir_center = np.arctan2(wy, wx)

        azmin, azmax, elmin, elmax = scan_range

        sim = AtmSim(
            azmin,
            azmax,
            elmin,
            elmax,
            tmin,
            tmax,
            self.lmin_center,
            self.lmin_sigma,
            self.lmax_center,
            self.lmax_sigma,
            w_center,
            0 * u.meter / u.second,
            wdir_center,
            0 * u.radian,
            self.z0_center,
            self.z0_sigma,
            T0_center,
            0 * u.Kelvin,
            self.zatm,
            self.zmax,
            xstep,
            ystep,
            zstep,
            self.nelem_sim_max,
            comm,
            key1,
            key2,
            counter1,
            counter2,
            cachedir,
            rmin * u.meter,
            rmax * u.meter,
            write_debug=self.debug_spectrum,
            node_comm=comm_node,
            node_rank_comm=comm_node_rank,
        )

        msg = f"{prefix}SimulateAtmosphere:  Initialize atmosphere"
        log.debug_rank(msg, comm=comm, timer=tmr)

        # Check if the cache already exists.

        use_cache = False
        have_cache = False
        if rank == 0:
            if cachedir is not None:
                # We are saving to cache
                use_cache = True
            fname = None
            if cachedir is not None:
                fname = os.path.join(
                    cachedir, "{}_{}_{}_{}.h5".format(key1, key2, counter1, counter2)
                )
                if os.path.isfile(fname):
                    if self.overwrite_cache:
                        os.remove(fname)
                    else:
                        have_cache = True
            if have_cache:
                log.debug(
                    f"{prefix}Loading the atmosphere for t = {tmin - tmin_tot} from {fname}"
                )
            else:
                log.debug(
                    f"{prefix}Simulating the atmosphere for t = {tmin - tmin_tot}"
                )
        if comm is not None:
            use_cache = comm.bcast(use_cache, root=0)

        err = sim.simulate(use_cache=use_cache)
        if err != 0:
            raise RuntimeError(prefix + "Simulation failed.")

        # Advance the sample counter in case wind_time broke the
        # observation in parts

        counter2 += 100000000

        op = None
        if have_cache:
            op = "Loaded"
        else:
            op = "Simulated"
        msg = f"{prefix}SimAtmosphere: {op} atmosphere"
        log.debug_rank(msg, comm=comm, timer=tmr)
        return sim, counter2

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [
                self.boresight,
            ],
            "detdata": list(),
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.turnaround_interval is not None:
            req["intervals"].append(self.turnaround_interval)
        return req

    def _provides(self):
        prov = {
            "global": [self.output],
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov


@trait_docs
class ObserveAtmosphere(Operator):
    """Operator which uses detector pointing to observe a simulated atmosphere slab."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for accumulating dipole timestreams",
    )

    quats_azel = Unicode(
        defaults.quats_azel,
        allow_none=True,
        help="Observation detdata key for detector quaternions",
    )

    weights = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for detector Stokes weights",
    )

    weights_mode = Unicode("IQU", help="Stokes weights mode (eg. 'I', 'IQU', 'QU')")

    polarization_fraction = Float(
        0,
        help="Polarization fraction (only Q polarization).",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of valid data in all observations"
    )

    wind_view = Unicode(
        "wind", help="The view of times matching individual simulated atmosphere slabs"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    sim = Unicode("atmsim", help="The observation key for the list of AtmSim objects")

    absorption = Unicode(
        None, allow_none=True, help="The observation key for the absorption"
    )

    loading = Unicode(None, allow_none=True, help="The observation key for the loading")

    n_bandpass_freqs = Int(
        100,
        help="The number of sampling frequencies used when convolving the bandpass "
        "with atmosphere absorption and loading",
    )

    sample_rate = Quantity(
        None,
        allow_none=True,
        help="Rate at which to sample atmospheric TOD before interpolation.  "
        "Default is no interpolation.",
    )

    fade_time = Quantity(
        None,
        allow_none=True,
        help="Fade in/out time to avoid a step at wind break.",
    )

    gain = Float(1.0, help="Scaling applied to the simulated TOD")

    debug_tod = Bool(False, help="If True, dump TOD to pickle files")

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        gt = GlobalTimers.get()

        for trait in ("absorption",):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        gt.start("ObserveAtmosphere:  total")

        comm = data.comm.comm_group
        group = data.comm.group
        rank = data.comm.group_rank

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            gt.start("ObserveAtmosphere:  per-observation setup")
            # Compute the (common) absorption and loading
            common_absorption, common_loading = self._common_absorption_and_loading(
                ob, comm
            )

            # Bandpass-specific unit conversion, relative to 150GHz
            absorption, loading = self._get_absorption_and_loading(
                ob, common_absorption, common_loading, dets
            )

            # Make sure detector data output exists
            exists = ob.detdata.ensure(self.det_data, detectors=dets)

            # Prefix for logging
            log_prefix = f"{group} : {ob.name}"

            # The current wind-driven timespan
            cur_wind = 0

            # Loop over views
            views = ob.view[self.view]

            ngood_tot = 0
            nbad_tot = 0

            gt.stop("ObserveAtmosphere:  per-observation setup")
            for vw in range(len(views)):
                times = views.shared[self.times][vw]

                # Determine the wind interval we are in, and hence which atmosphere
                # simulation to use.  The wind intervals are already guaranteed
                # by the calling code to break on the data view boundaries.
                if len(views) > 1:
                    while (
                        cur_wind < (len(ob.view[self.wind_view]) - 1)
                        and times[0]
                        > ob.view[self.wind_view].shared[self.times][cur_wind][-1]
                    ):
                        cur_wind += 1

                # Get the flags if needed
                sh_flags = None
                if self.shared_flags is not None:
                    sh_flags = (
                        np.array(views.shared[self.shared_flags][vw])
                        & self.shared_flag_mask
                    )

                sim_list = ob[self.sim][cur_wind]

                for det in dets:
                    gt.start("ObserveAtmosphere:  detector setup")
                    flags = None
                    if self.det_flags is not None:
                        flags = (
                            np.array(views.detdata[self.det_flags][vw][det])
                            & self.det_flag_mask
                        )
                        if sh_flags is not None:
                            flags |= sh_flags
                    elif sh_flags is not None:
                        flags = sh_flags

                    good = slice(None, None, None)
                    ngood = len(views.detdata[self.det_data][vw][det])
                    if flags is not None:
                        good = flags == 0
                        ngood = np.sum(good)

                    if ngood == 0:
                        continue
                    ngood_tot += ngood

                    # Detector Az / El quaternions for good samples
                    azel_quat = views.detdata[self.quats_azel][vw][det][good]

                    # Convert Az/El quaternion of the detector back into
                    # angles from the simulation.
                    theta, phi = qa.to_position(azel_quat)

                    # Stokes weights for observing polarized atmosphere
                    if self.weights is None:
                        weights_I = 1
                        weights_Q = 0
                    else:
                        weights = views.detdata[self.weights][vw][det][good]
                        if "I" in self.weights_mode:
                            ind = self.weights_mode.index("I")
                            weights_I = weights[:, ind].copy()
                        else:
                            weights_I = 0
                        if "Q" in self.weights_mode:
                            ind = self.weights_mode.index("Q")
                            weights_Q = weights[:, ind].copy()
                        else:
                            weights_Q = 0

                    # Azimuth is measured in the opposite direction
                    # than longitude
                    az = 2 * np.pi - phi
                    el = np.pi / 2 - theta

                    if np.ptp(az) < np.pi:
                        azmin_det = np.amin(az)
                        azmax_det = np.amax(az)
                    else:
                        # Scanning across the zero azimuth.
                        azmin_det = np.amin(az[az > np.pi]) - 2 * np.pi
                        azmax_det = np.amax(az[az < np.pi])
                    elmin_det = np.amin(el)
                    elmax_det = np.amax(el)

                    tmin_det = times[good][0]
                    tmax_det = times[good][-1]

                    # We may be interpolating some of the time samples

                    if self.sample_rate is None:
                        t_interp = times[good]
                        az_interp = az
                        el_interp = el
                    else:
                        n_interp = int(
                            (tmax_det - tmin_det) * self.sample_rate.to_value(u.Hz)
                        )
                        t_interp = np.linspace(tmin_det, tmax_det, n_interp)
                        # Az is discontinuous if we scan across az=0.  To interpolate,
                        # we must unwrap it first ...
                        az_interp = np.interp(t_interp, times[good], np.unwrap(az))
                        # ... however, the checks later assume 0 < az < 2pi
                        az_interp[az_interp < 0] += 2 * np.pi
                        az_interp[az_interp > 2 * np.pi] -= 2 * np.pi
                        el_interp = np.interp(t_interp, times[good], el)

                    # Integrate detector signal across all slabs at different altitudes

                    atmdata = np.zeros(t_interp.size, dtype=np.float64)

                    gt.stop("ObserveAtmosphere:  detector setup")
                    gt.start("ObserveAtmosphere:  detector AtmSim.observe")
                    for icur, cur_sim in enumerate(sim_list):
                        if cur_sim.tmin > tmin_det or cur_sim.tmax < tmax_det:
                            msg = (
                                f"{log_prefix} : {det} "
                                f"Detector time: [{tmin_det:.1f}, {tmax_det:.1f}], "
                                f"is not contained in [{cur_sim.tmin:.1f}, "
                                f"{cur_sim.tmax:.1f}]"
                            )
                            raise RuntimeError(msg)
                        if (
                            not (
                                cur_sim.azmin <= azmin_det
                                and azmax_det <= cur_sim.azmax
                            )
                            and not (
                                cur_sim.azmin <= azmin_det - 2 * np.pi
                                and azmax_det - 2 * np.pi <= cur_sim.azmax
                            )
                        ) or not (
                            cur_sim.elmin <= elmin_det and elmin_det <= cur_sim.elmax
                        ):
                            msg = (
                                f"{log_prefix} : {det} "
                                f"Detector Az/El: [{azmin_det:.5f}, {azmax_det:.5f}], "
                                f"[{elmin_det:.5f}, {elmax_det:.5f}] is not contained "
                                f"in [{cur_sim.azmin:.5f}, {cur_sim.azmax:.5f}], "
                                f"[{cur_sim.elmin:.5f} {cur_sim.elmax:.5f}]"
                            )
                            raise RuntimeError(msg)

                        err = cur_sim.observe(
                            t_interp, az_interp, el_interp, atmdata, -1.0
                        )

                        # Dump timestream snapshot
                        if self.debug_tod:
                            first = ob.intervals[self.view][vw].first
                            last = ob.intervals[self.view][vw].last
                            self._save_tod(
                                f"post{icur}",
                                ob,
                                self.times,
                                first,
                                last,
                                det,
                                raw=atmdata,
                            )

                        if err != 0:
                            # import pdb
                            # import matplotlib.pyplot as plt
                            # pdb.set_trace()
                            # Observing failed
                            if self.sample_rate is None:
                                full_data = atmdata
                            else:
                                # Interpolate to full sample rate, make sure to flag
                                # samples around a failed time sample
                                test_data = atmdata.copy()
                                for i in [-2, -1, 1, 2]:
                                    test_data *= np.roll(atmdata, i)
                                interp = scipy.interpolate.interp1d(
                                    t_interp,
                                    test_data,
                                    kind="previous",
                                    copy=False,
                                )
                                full_data = interp(times[good])
                            bad = np.abs(full_data) < 1e-30
                            nbad = np.sum(bad)
                            log.error(
                                f"{log_prefix} : {det} "
                                f"ObserveAtmosphere failed for {nbad} "
                                f"({nbad * 100 / ngood:.2f} %) samples.  "
                                f"det = {det}, rank = {rank}"
                            )
                            # If any samples failed the simulation, flag them as bad
                            if nbad > 0:
                                if self.det_flags is None:
                                    log.warning(
                                        "Some samples failed atmosphere simulation, "
                                        "but no det flag field was specified.  "
                                        "Cannot flag samples"
                                    )
                                else:
                                    views.detdata[self.det_flags][vw][det][good][
                                        bad
                                    ] |= self.det_flag_mask
                                    nbad_tot += nbad
                    gt.stop("ObserveAtmosphere:  detector AtmSim.observe")

                    # Optionally, interpolate the atmosphere to full sample rate
                    if self.sample_rate is not None:
                        gt.start("ObserveAtmosphere:  detector interpolate")
                        interp = scipy.interpolate.interp1d(
                            t_interp,
                            atmdata,
                            kind="quadratic",
                            copy=False,
                        )
                        atmdata = interp(times[good])
                        gt.stop("ObserveAtmosphere:  detector interpolate")

                    gt.start("ObserveAtmosphere:  detector accumulate")

                    # Dump timestream snapshot
                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "precal", ob, self.times, first, last, det, raw=atmdata
                        )

                    # Calibrate the atmospheric fluctuations to appropriate bandpass
                    atmdata *= self.gain * absorption[det]

                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "postcal", ob, self.times, first, last, det, raw=atmdata
                        )

                    # If we are simulating disjoint wind views, we need to suppress
                    # a jump between them

                    if len(views) > 1 and self.fade_time is not None:
                        atmdata -= np.mean(atmdata)  # Add thermal loading after this
                        fsample = (times.size - 1) / (times[-1] - times[0])
                        nfade = min(
                            int(self.fade_time.to_value(u.s) * fsample),
                            atmdata.size // 2,
                        )
                        if vw < len(views) - 1:
                            # Fade out the end
                            atmdata[-nfade:] *= np.arange(nfade - 1, -1, -1) / nfade
                        if vw > 0:
                            # Fade out the beginning
                            atmdata[:nfade] *= np.arange(nfade) / nfade

                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "postfade", ob, self.times, first, last, det, raw=atmdata
                        )

                    # Add polarization.  In our simple model, there is only Q-polarization
                    # and the polarization fraction is constant.
                    pfrac = self.polarization_fraction
                    atmdata *= weights_I + weights_Q * pfrac

                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "postpol", ob, self.times, first, last, det, raw=atmdata
                        )

                    if loading is not None:
                        # Add the elevation-dependent atmospheric loading
                        atmdata += loading[det] / np.sin(el)

                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "postload", ob, self.times, first, last, det, raw=atmdata
                        )

                    # Add contribution to output
                    views.detdata[self.det_data][vw][det, good] += atmdata
                    gt.stop("ObserveAtmosphere:  detector accumulate")

                    # Dump timestream snapshot
                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "final",
                            ob,
                            self.times,
                            first,
                            last,
                            det,
                            detdata=self.det_data,
                        )

            if nbad_tot > 0:
                frac = nbad_tot / (ngood_tot + nbad_tot) * 100
                log.error(
                    f"{log_prefix}: Observe atmosphere FAILED on {frac:.2f}% of samples"
                )
        gt.stop("ObserveAtmosphere:  total")

    @function_timer
    def _common_absorption_and_loading(self, obs, comm):
        """Compute the (common) absorption and loading prior to bandpass convolution.

        Args:
            obs (Observation):  One observation in the session.
            comm (MPI.Comm):  Optional communicator over which to distribute the
                calculations.

        Returns:
            (tuple):  The absorption, loading vectors for the observation.

        """

        if obs.telescope.focalplane.bandpass is None:
            raise RuntimeError("Focalplane does not define bandpass")
        altitude = obs.telescope.site.earthloc.height
        weather = obs.telescope.site.weather
        bandpass = obs.telescope.focalplane.bandpass

        freq_min, freq_max = bandpass.get_range()
        n_freq = self.n_bandpass_freqs
        freqs = np.linspace(freq_min, freq_max, n_freq)
        if comm is None:
            ntask = 1
            my_rank = 0
        else:
            ntask = comm.size
            my_rank = comm.rank
        n_freq_task = int(np.ceil(n_freq / ntask))
        my_start = min(my_rank * n_freq_task, n_freq)
        my_stop = min(my_start + n_freq_task, n_freq)
        my_n_freq = my_stop - my_start

        if my_n_freq > 0:
            absorption = atm_absorption_coefficient_vec(
                altitude.to_value(u.meter),
                weather.air_temperature.to_value(u.Kelvin),
                weather.surface_pressure.to_value(u.Pa),
                weather.pwv.to_value(u.mm),
                freqs[my_start].to_value(u.GHz),
                freqs[my_stop - 1].to_value(u.GHz),
                my_n_freq,
            )
            loading = atm_atmospheric_loading_vec(
                altitude.to_value(u.meter),
                weather.air_temperature.to_value(u.Kelvin),
                weather.surface_pressure.to_value(u.Pa),
                weather.pwv.to_value(u.mm),
                freqs[my_start].to_value(u.GHz),
                freqs[my_stop - 1].to_value(u.GHz),
                my_n_freq,
            )
        else:
            absorption, loading = [], []

        if comm is not None:
            absorption = np.hstack(comm.allgather(absorption))
            loading = np.hstack(comm.allgather(loading))
        return (absorption, loading)

    @function_timer
    def _detector_absorption_and_loading(self, obs, absorption, loading, dets):
        """Bandpass-specific unit conversion and loading"""

        if obs.telescope.focalplane.bandpass is None:
            raise RuntimeError("Focalplane does not define bandpass")
        bandpass = obs.telescope.focalplane.bandpass

        freq_min, freq_max = bandpass.get_range()
        n_freq = self.n_bandpass_freqs
        freqs = np.linspace(freq_min, freq_max, n_freq)

        absorption_det = {}
        for det in dets:
            absorption_det[det] = bandpass.convolve(det, freqs, absorption, rj=True)

        if loading is None:
            loading_det = None
        else:
            loading_det = {}
            for det in dets:
                loading_det[det] = bandpass.convolve(det, freqs, loading, rj=True)
        return absorption_det, loading_det

    @function_timer
    def _save_tod(
        self,
        prefix,
        ob,
        times,
        first,
        last,
        det,
        raw=None,
        detdata=None,
    ):
        import pickle

        outdir = "snapshots"
        try:
            os.makedirs(outdir)
        except FileExistsError:
            pass

        timestamps = ob.shared[times].data
        tmin = int(timestamps[first])
        tmax = int(timestamps[last])
        slc = slice(first, last + 1, 1)

        ddata = None
        if raw is not None:
            ddata = raw
        else:
            ddata = ob.detdata[detdata][det, slc]

        fn = os.path.join(
            outdir,
            f"atm_tod_{prefix}_{ob.name}_{det}_t_{tmin}_{tmax}.pck",
        )
        with open(fn, "wb") as fout:
            pickle.dump([det, timestamps[slc], ddata], fout)
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [self.sim],
            "shared": [
                self.times,
            ],
            "detdata": [
                self.det_data,
                self.quats_azel,
            ],
            "intervals": [
                self.wind_view,
            ],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.weights is not None:
            req["weights"].append(self.weights)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [
                self.det_data,
            ],
            "intervals": list(),
        }
        return prov
