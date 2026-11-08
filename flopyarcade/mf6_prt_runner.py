#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
MODFLOW 6 + PRT (Particle Tracking) runner for FloPyArcade.

Replaces MF2005+MODPATH with MF6 GWF + PRT particle tracking in one simulation.
Both models run coupled via the GWF6-PRT6 exchange (no external budget file needed).

Execution uses the BMI (Basic Model Interface) via libmf6.so + xmipy for
in-process, per-step control — no subprocess spawn, no file I/O per call.
"""

import os
import csv
import shutil
import numpy as np
import pandas as pd
from os.path import abspath, dirname, exists, join

from flopy.mf6 import (
    MFSimulation, ModflowTdis, ModflowIms, ModflowEms,
    ModflowGwf, ModflowGwfdis, ModflowGwfic, ModflowGwfchd,
    ModflowGwfnpf, ModflowGwfoc, ModflowGwfsto, ModflowGwfrcha,
    ModflowGwfwel,
    ModflowPrt, ModflowPrtdis, ModflowPrtprp, ModflowPrtoc, ModflowPrtmip,
    ModflowGwfprt,
)


class MF6PRTMixin:
    """Mixin providing MF6+PRT simulation via BMI as alternative to MF2005+MODPATH.

    When solver_backend='mf6_prt', _step_run_simulation dispatches here.
    Builds a coupled GWF6+PRT6 simulation, executes via BMI (libmf6.so),
    parses track CSV into the same DataFrame format the MODPATH pipeline expects.
    """

    def _mf6prt_get_exe(self):
        """Get MF6 executable path (for file-based fallback)."""
        base = abspath(dirname(__file__))
        exe = join(base, 'simulators', 'linux', 'mf6')
        if not exists(exe):
            exe = join('simulators', 'linux', 'mf6')
        return abspath(exe)

    def _mf6prt_get_lib(self):
        """Get libmf6.so path for BMI execution."""
        base = abspath(dirname(__file__))
        lib = join(base, 'simulators', 'linux', 'libmf6.so')
        if not exists(lib):
            # Fallback: search common locations
            for candidate in [
                join('simulators', 'linux', 'libmf6.so'),
                '/usr/lib/libmf6.so',
                '/usr/local/lib/libmf6.so',
            ]:
                if exists(candidate):
                    lib = abspath(candidate)
                    break
            else:
                raise FileNotFoundError(
                    f"libmf6.so not found. Looked in: {join(base, 'simulators', 'linux')}. "
                    f"Install MF6 6.7.0+ with BMI support."
                )
        return abspath(lib)

    def _mf6prt_cell_from_coords(self, x, y):
        """Convert model-space (x, y) to 0-based (row, col).

        FloPyArcade: x east (col), y north. Row 0 = NORTH in MF6.
        MF6 row 1 (1-based) is at the top (highest y).
        """
        col0b = int(x / self.dCol)
        row0b = self.nRow - 1 - int(y / self.dRow)
        col0b = max(0, min(col0b, self.nCol - 1))
        row0b = max(0, min(row0b, self.nRow - 1))
        return row0b, col0b

    def _mf6prt_get_wells(self):
        """Build MF6 WEL data from environment well attributes.

        Returns list of ((layer, row, col), Q) tuples (0-based).
        """
        wells = []
        if hasattr(self, 'wellCellLayer') and hasattr(self, 'wellQ'):
            l = self.wellCellLayer - 1
            r = self.wellCellRow - 1
            c = self.wellCellColumn - 1
            wells.append(((l, r, c), self.wellQ))
        n_hw = getattr(self, 'nHelperWells', 0)
        if n_hw > 0 and hasattr(self, 'helperWells'):
            for i in range(n_hw):
                idx = str(i + 1)
                hl = self.helperWells[f'l{idx}'] - 1
                hr = self.helperWells[f'r{idx}'] - 1
                hc = self.helperWells[f'c{idx}'] - 1
                hq = self.helperWells[f'wellQ{idx}']
                wells.append(((hl, hr, hc), hq))
        return wells

    def _mf6prt_get_hydraulics(self):
        """Get hydraulic parameters. Handles both 0s-c and general env attribute names."""
        hk = getattr(self, 'hk', None)
        if hk is None or hk == 0.0:
            hk = 10.0

        vka = 10.0
        ss = 1e-5
        sy = 0.15

        chd_west = getattr(self, 'chd_west', None)
        chd_east = getattr(self, 'chd_east', None)
        if chd_west is None:
            chd_west = getattr(self, 'headSpecWest', None)
        if chd_east is None:
            chd_east = getattr(self, 'headSpecEast', None)

        rch = getattr(self, 'rch_rate', None) or getattr(self, 'recharge', None)

        return hk, vka, ss, sy, chd_west, chd_east, rch

    def _mf6prt_build_model(self, sim_name, workdir):
        """Build the GWF6+PRT6 coupled simulation model (files only, no execution).

        Uses multi-period TDIS (nper=NSTP, nstp=1) for BMI per-step access.
        """
        exe = self._mf6prt_get_exe()
        nlay, nrow, ncol = self.nLay, self.nRow, self.nCol
        delr, delc = self.dRow, self.dCol
        top = self.zTop
        nstp = self.periodSteps

        sim = MFSimulation(
            sim_name=sim_name, version='mf6', exe_name=exe, sim_ws=workdir
        )

        # Multi-period TDIS: each period is 1 timestep of 1 day
        # Required for BMI per-step access (single-period TDIS causes do_time_step to run all steps)
        tdis = ModflowTdis(
            sim, time_units='DAYS', nper=nstp,
            perioddata=[(1, 1.0, 1.0)] * nstp
        )

        # === GWF MODEL ===
        gwf = ModflowGwf(sim, modelname='gwf', save_flows=True)
        ModflowGwfdis(
            gwf, nlay=nlay, nrow=nrow, ncol=ncol,
            delr=delr, delc=delc, top=top, botm=[0.0] * nlay
        )

        # IC: steady-state gradient or full spatial array
        use_steady_ic = self.env_config.get('steady_state_ic', False)
        if use_steady_ic:
            h_west = getattr(self, 'headSpecWest', 60.0)
            h_east = getattr(self, 'headSpecEast', 56.0)
            cols = np.arange(ncol, dtype=float)
            h_profile = h_west + (h_east - h_west) * cols / (ncol - 1)
            strt_arr = np.tile(h_profile, (nlay, nrow, 1))
        else:
            strt_val = self.strt
            if isinstance(strt_val, (np.ndarray, list)):
                strt_arr = np.asarray(strt_val, dtype=float)
            else:
                strt_arr = float(strt_val)
        ModflowGwfic(gwf, strt=strt_arr)

        # Get hydraulics
        hk, vka, ss, sy, chd_west, chd_east, rch = self._mf6prt_get_hydraulics()

        # NPF
        ModflowGwfnpf(
            gwf, icelltype=1, k=hk, save_flows=True
        )

        # STO
        if ss > 0:
            ModflowGwfsto(gwf, iconvert=0, ss=ss, sy=sy)

        # WEL (all periods)
        wells = self._mf6prt_get_wells()
        if wells:
            wel_spd = {}
            for i in range(nstp):
                wel_spd[i] = [((l, r, c), q) for ((l, r, c), q) in wells]
            ModflowGwfwel(
                gwf, maxbound=len(wells),
                stress_period_data=wel_spd, save_flows=True
            )

        # RCH
        if rch is not None and rch > 0:
            rch_array = np.full((nlay, nrow, ncol), float(rch))
            ModflowGwfrcha(gwf, recharge=rch_array, save_flows=True)

        # CHD (all periods)
        chd_rec = []
        for r in range(nrow):
            if chd_west is not None:
                chd_rec.append([(0, r, 0), float(chd_west)])
            if chd_east is not None:
                chd_rec.append([(0, r, ncol - 1), float(chd_east)])
        if chd_rec:
            chd_spd = {i: chd_rec for i in range(nstp)}
            ModflowGwfchd(
                gwf, maxbound=len(chd_rec),
                stress_period_data=chd_spd, save_flows=True
            )

        # OC (heads for all periods)
        ModflowGwfoc(
            gwf, head_filerecord=f'{sim_name}.hds',
            saverecord=[('HEAD', 'ALL')]
        )

        # IMS for GWF (MUST be registered before EMS for PRT)
        ims = ModflowIms(
            sim, outer_dvclose=1e-6, inner_dvclose=1e-6,
            outer_maximum=500, inner_maximum=100
        )
        sim.register_ims_package(ims, ['gwf'])

        # === PRT MODEL ===
        prt = ModflowPrt(sim, modelname='prt')
        ModflowPrtdis(
            prt, nlay=nlay, nrow=nrow, ncol=ncol,
            delr=delr, delc=delc, top=top, botm=[0.0] * nlay
        )
        ModflowPrtmip(prt, porosity=0.2)

        # Release point at current particle position
        px, py, pz = self.particleCoords
        row0b, col0b = self._mf6prt_cell_from_coords(px, py)
        release_data = [(0, 0, row0b, col0b, float(px), float(py), float(pz))]

        ModflowPrtprp(
            prt, extend_tracking=True,
            track_filerecord=f'{sim_name}.trk',
            trackcsv_filerecord=f'{sim_name}.trk.csv',
            nreleasepts=1, packagedata=release_data
        )
        # IMPORTANT: ModflowPrtoc must NOT duplicate track_filerecord (causes file-open crash)
        ModflowPrtoc(prt)

        # EMS for PRT
        ems = ModflowEms(sim)
        sim.register_solution_package(ems, ['prt'])

        # GWF6-PRT6 exchange
        ModflowGwfprt(sim, exgtype='GWF6-PRT6', exgmnamea='gwf', exgmnameb='prt')

        # Write all model files
        sim.write_simulation(silent=True)
        return sim_name

    def _mf6prt_run_bmi(self, sim_name, workdir):
        """Execute the model via BMI (libmf6.so) with per-step control.

        Returns None on success; raises RuntimeError on failure.
        """
        from xmipy import XmiWrapper

        libmf6 = self._mf6prt_get_lib()
        nam_file = f'{sim_name}.nam'

        old_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            bmi = XmiWrapper(libmf6)
            bmi.initialize(nam_file)

            nstp = self.periodSteps
            for step in range(nstp):
                dt = bmi.get_time_step()
                if dt == 0.0:
                    dt = 1.0  # fallback for first step
                bmi.prepare_time_step(dt)
                bmi.do_time_step()

            bmi.finalize()
        finally:
            os.chdir(old_cwd)

    def _mf6prt_run_subprocess(self, sim_name, workdir):
        """Execute the model via subprocess (fallback when libmf6.so unavailable)."""
        exe = self._mf6prt_get_exe()
        nam_file = f'{sim_name}.nam'
        import subprocess
        r = subprocess.run(
            [exe, nam_file], cwd=workdir,
            capture_output=True, timeout=120
        )
        if r.returncode != 0:
            lst = join(workdir, 'mfsim.lst')
            if exists(lst):
                with open(lst) as f:
                    content = f.read()
                    for line in content.splitlines():
                        if 'ERROR' in line:
                            raise RuntimeError(f"MF6+PRT failed: {line.strip()}")
            raise RuntimeError(
                f"MF6+PRT subprocess failed rc={r.returncode}: {r.stderr.decode()[:200]}"
            )

    def _mf6prt_build_and_run(self, workdir):
        """Build and run one MF6+PRT step. Returns DataFrame with x,y,z,time.

        Tries BMI first; falls back to subprocess if libmf6.so is unavailable.
        """
        if os.path.exists(workdir):
            shutil.rmtree(workdir)
        os.makedirs(workdir, exist_ok=True)

        sim_name = f"{self.MODELNAME}_prt"
        self._mf6prt_build_model(sim_name, workdir)

        # Execute: prefer BMI, fallback to subprocess
        try:
            self._mf6prt_run_bmi(sim_name, workdir)
        except (ImportError, FileNotFoundError) as e:
            # BMI unavailable, use subprocess
            self._mf6prt_run_subprocess(sim_name, workdir)

        # Parse track CSV (written by both BMI and subprocess paths)
        track_csv = join(workdir, f'{sim_name}.trk.csv')
        if not exists(track_csv):
            import glob
            candidates = glob.glob(join(workdir, '*.trk.csv'))
            if candidates:
                track_csv = candidates[0]
            else:
                raise RuntimeError(f"Track CSV not found in {workdir}")

        return self._mf6prt_parse_track_csv(track_csv)

    def _mf6prt_parse_track_csv(self, csv_path):
        """Parse PRT track CSV into DataFrame with x, y, z, time columns."""
        xs, ys, zs, ts = [], [], [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    xs.append(float(row['x']))
                    ys.append(float(row['y']))
                    zs.append(float(row['z']))
                    ts.append(float(row['t']))
                except (ValueError, KeyError):
                    continue
        if not xs:
            raise RuntimeError(f"Track CSV is empty: {csv_path}")
        return pd.DataFrame({'x': xs, 'y': ys, 'z': zs, 'time': ts})

    def _step_run_simulation_mf6prt(self):
        """Pipeline stage 2 (MF6+PRT): run coupled flow+tracking, update trajectories.

        Handles trajectory storage and coordinate update inline with numpy arrays
        to avoid pandas label-based indexing issues in reward.py.
        """
        from copy import copy
        workdir = join(self.modelpth, 'mf6prt_run')
        data = self._mf6prt_build_and_run(workdir)

        # Filter by time (uses env_base._filterByTime - works with DataFrame)
        valid = self._filterByTime(data)

        # Append trajectories as numpy arrays (positional [-1] works in reward.py)
        for coord in 'xyz':
            series = data[coord]
            arr = np.asarray(series[valid].values) if valid is not None else np.asarray(series.values)
            self.trajectories[coord].append(arr)

        # Update particle coordinates to last valid position
        coords = [float(np.asarray(data[c][valid].values)[-1]) for c in 'xyz']
        self.particleCoords = copy(coords)

        # Cleanup
        shutil.rmtree(workdir, ignore_errors=True)
