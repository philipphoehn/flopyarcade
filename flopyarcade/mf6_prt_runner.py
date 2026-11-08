#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
MODFLOW 6 + PRT (Particle Tracking) runner for FloPyArcade.

Replaces MF2005+MODPATH with MF6 GWF + PRT particle tracking in one simulation.
Both models run coupled via the GWF6-PRT6 exchange (no external budget file needed).

Execution uses the BMI (Basic Model Interface) via libmf6.so + xmipy for
in-process, per-step control — no subprocess spawn, no file I/O per call.

Persistent session: build+init once per episode, then each RL step just
advances the BMI timestep and reads particle position in-memory.
"""

import os
import csv
import time as _time
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
    reads particle positions in-memory (no file I/O for output).

    Persistent session mode: model built+initialized once per episode,
    each step advances the BMI and reads particle — no rebuild.
    """

    # ──────────────────────────────────────────────
    # Binary / library paths
    # ──────────────────────────────────────────────

    def _mf6prt_get_exe(self):
        """Get MF6 executable path."""
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

    # ──────────────────────────────────────────────
    # Model data helpers
    # ──────────────────────────────────────────────

    def _mf6prt_cell_from_coords(self, x, y):
        """Convert model-space (x, y) to 0-based (row, col)."""
        col0b = int(x / self.dCol)
        row0b = self.nRow - 1 - int(y / self.dRow)
        col0b = max(0, min(col0b, self.nCol - 1))
        row0b = max(0, min(row0b, self.nRow - 1))
        return row0b, col0b

    def _mf6prt_get_wells(self):
        """Build MF6 WEL data from environment well attributes."""
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
        """Get hydraulic parameters."""
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

    # ──────────────────────────────────────────────
    # Model build (shared between persistent and one-shot)
    # ──────────────────────────────────────────────

    def _mf6prt_build_model_files(self, sim_name, workdir, nper):
        """Build GWF6+PRT6 coupled simulation model files.

        Uses nper=1, nstp=nper (single period, many timesteps) to avoid
        Python overhead of building per-period stress data structures.

        Args:
            sim_name: simulation name
            workdir: directory to write files
            nper: total number of timesteps (used as nstp in single period)
        """
        exe = self._mf6prt_get_exe()
        nlay, nrow, ncol = self.nLay, self.nRow, self.nCol
        delr, delc = self.dRow, self.dCol
        top = self.zTop

        sim = MFSimulation(
            sim_name=sim_name, version='mf6', exe_name=exe, sim_ws=workdir
        )

        # Single period with many timesteps (avoids per-period Python overhead)
        tdis = ModflowTdis(
            sim, time_units='DAYS', nper=1,
            perioddata=[(nper, 1.0, 1.0)]
        )

        # === GWF MODEL ===
        gwf = ModflowGwf(sim, modelname='gwf', save_flows=True)
        ModflowGwfdis(
            gwf, nlay=nlay, nrow=nrow, ncol=ncol,
            delr=delr, delc=delc, top=top, botm=[0.0] * nlay
        )

        # IC
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

        # Hydraulics
        hk, vka, ss, sy, chd_west, chd_east, rch = self._mf6prt_get_hydraulics()
        ModflowGwfnpf(gwf, icelltype=1, k=hk, save_flows=True)

        if ss > 0:
            ModflowGwfsto(gwf, iconvert=0, ss=ss, sy=sy)

        # WEL (single period, all timesteps same)
        wells = self._mf6prt_get_wells()
        if wells:
            wel_spd = {0: [((l, r, c), q) for ((l, r, c), q) in wells]}
            ModflowGwfwel(
                gwf, maxbound=len(wells),
                stress_period_data=wel_spd, save_flows=True
            )

        # RCH
        if rch is not None and rch > 0:
            rch_array = np.full((nlay, nrow, ncol), float(rch))
            ModflowGwfrcha(gwf, recharge=rch_array, save_flows=True)

        # CHD (all periods same)
        chd_rec = []
        for r in range(nrow):
            if chd_west is not None:
                chd_rec.append([(0, r, 0), float(chd_west)])
            if chd_east is not None:
                chd_rec.append([(0, r, ncol - 1), float(chd_east)])
        if chd_rec:
            chd_spd = {0: chd_rec}
            ModflowGwfchd(
                gwf, maxbound=len(chd_rec),
                stress_period_data=chd_spd, save_flows=True
            )

        # OC (heads)
        ModflowGwfoc(
            gwf, head_filerecord=f'{sim_name}.hds',
            saverecord=[('HEAD', 'ALL')]
        )

        # IMS for GWF
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
            nreleasepts=1, packagedata=release_data
        )
        ModflowPrtoc(prt)

        # EMS for PRT
        ems = ModflowEms(sim)
        sim.register_solution_package(ems, ['prt'])

        # GWF6-PRT6 exchange
        ModflowGwfprt(sim, exgtype='GWF6-PRT6', exgmnamea='gwf', exgmnameb='prt')

        # Write all model files
        sim.write_simulation(silent=True)

    # ──────────────────────────────────────────────
    # PERSISTENT BMI SESSION (primary path)
    # ──────────────────────────────────────────────

    def _mf6prt_persistent_init(self, workdir):
        """Initialize persistent BMI session for the entire episode.

        Builds model with nper = maxSteps * periodSteps periods,
        writes files, loads libmf6.so, and initializes BMI.
        Called once at episode start (reset).
        """
        from xmipy import XmiWrapper

        t0 = _time.perf_counter()

        if os.path.exists(workdir):
            shutil.rmtree(workdir)
        os.makedirs(workdir, exist_ok=True)

        # Total MF6 periods for the full episode
        total_periods = self.maxSteps * self.periodSteps
        sim_name = f"{self.MODELNAME}_prt_persist"

        # Build model files
        self._mf6prt_build_model_files(sim_name, workdir, total_periods)

        # Initialize BMI
        libmf6 = self._mf6prt_get_lib()
        nam_file = os.path.join(workdir, f'{sim_name}.nam')

        old_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            self._bmi = XmiWrapper(libmf6)
            self._bmi.initialize(f'{sim_name}.nam')
        finally:
            os.chdir(old_cwd)

        self._bmi_workdir = workdir
        self._bmi_step = 0  # how many periods advanced so far
        self._bmi_total = total_periods
        self._bmi_init_time = _time.perf_counter() - t0

    def _mf6prt_persistent_step(self):
        """Advance one RL step (periodSteps BMI periods) and read particle.

        Returns DataFrame with x, y, z, time columns (periodSteps+1 rows:
        initial position at t=0 plus periodSteps positions).
        """
        bmi = self._bmi
        if bmi is None:
            raise RuntimeError("BMI session not initialized. Call _mf6prt_persistent_init first.")

        nstp = self.periodSteps
        xs, ys, zs, ts = [], [], [], []

        # Record starting position (known from previous step or initial)
        pc = self.particleCoords
        xs.append(float(pc[0]))
        ys.append(float(pc[1]))
        zs.append(float(pc[2]))
        ts.append(0.0)

        old_cwd = os.getcwd()
        os.chdir(self._bmi_workdir)
        try:
            for _ in range(nstp):
                if self._bmi_step >= self._bmi_total:
                    break
                dt = bmi.get_time_step()
                if dt == 0.0:
                    dt = 1.0
                bmi.prepare_time_step(dt)
                bmi.do_time_step()
                self._bmi_step += 1

                # Read particle position in-memory
                x_val = bmi.get_value('PRT/PRP_0/PLX')
                y_val = bmi.get_value('PRT/PRP_0/PLY')
                z_val = bmi.get_value('PRT/PRP_0/PLZ')

                x_flat = float(np.asarray(x_val).flatten()[0])
                y_flat = float(np.asarray(y_val).flatten()[0])
                z_flat = float(np.asarray(z_val).flatten()[0])

                if not np.isnan(x_flat) and not np.isnan(y_flat):
                    xs.append(x_flat)
                    ys.append(y_flat)
                    zs.append(z_flat)
                    ts.append(bmi.get_current_time())
        finally:
            os.chdir(old_cwd)

        if len(xs) < 2:
            raise RuntimeError("BMI PRT returned no valid particle positions")

        return pd.DataFrame({'x': xs, 'y': ys, 'z': zs, 'time': ts})

    def _mf6prt_persistent_close(self):
        """Finalize and clean up the persistent BMI session."""
        if hasattr(self, '_bmi') and self._bmi is not None:
            try:
                old_cwd = os.getcwd()
                os.chdir(self._bmi_workdir)
                try:
                    self._bmi.finalize()
                finally:
                    os.chdir(old_cwd)
            except Exception:
                pass
            self._bmi = None

        # Cleanup workdir
        if hasattr(self, '_bmi_workdir') and self._bmi_workdir and os.path.exists(self._bmi_workdir):
            shutil.rmtree(self._bmi_workdir, ignore_errors=True)
            self._bmi_workdir = None

    # ──────────────────────────────────────────────
    # ONE-SHOT (fallback when BMI unavailable)
    # ──────────────────────────────────────────────

    def _mf6prt_run_bmi_oneshot(self, sim_name, workdir):
        """Execute one MF6+PRT step via BMI (non-persistent)."""
        from xmipy import XmiWrapper
        libmf6 = self._mf6prt_get_lib()
        nstp = self.periodSteps

        old_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            bmi = XmiWrapper(libmf6)
            bmi.initialize(f'{sim_name}.nam')

            xs, ys, zs, ts = [], [], [], []
            pc = self.particleCoords
            xs.append(float(pc[0]))
            ys.append(float(pc[1]))
            zs.append(float(pc[2]))
            ts.append(0.0)

            for step in range(nstp):
                dt = bmi.get_time_step()
                if dt == 0.0:
                    dt = 1.0
                bmi.prepare_time_step(dt)
                bmi.do_time_step()

                x_val = bmi.get_value('PRT/PRP_0/PLX')
                y_val = bmi.get_value('PRT/PRP_0/PLY')
                z_val = bmi.get_value('PRT/PRP_0/PLZ')

                x_flat = float(np.asarray(x_val).flatten()[0])
                y_flat = float(np.asarray(y_val).flatten()[0])
                z_flat = float(np.asarray(z_val).flatten()[0])

                if not np.isnan(x_flat) and not np.isnan(y_flat):
                    xs.append(x_flat)
                    ys.append(y_flat)
                    zs.append(z_flat)
                    ts.append(bmi.get_current_time())

            bmi.finalize()
        finally:
            os.chdir(old_cwd)

        if not xs:
            raise RuntimeError("BMI PRT returned no valid particle positions")
        return pd.DataFrame({'x': xs, 'y': ys, 'z': zs, 'time': ts})

    def _mf6prt_run_subprocess(self, sim_name, workdir):
        """Execute model via subprocess (fallback)."""
        exe = self._mf6prt_get_exe()
        import subprocess
        r = subprocess.run(
            [exe, f'{sim_name}.nam'], cwd=workdir,
            capture_output=True, timeout=120
        )
        if r.returncode != 0:
            lst = join(workdir, 'mfsim.lst')
            if exists(lst):
                with open(lst) as f:
                    for line in f.read().splitlines():
                        if 'ERROR' in line:
                            raise RuntimeError(f"MF6+PRT failed: {line.strip()}")
            raise RuntimeError(f"MF6+PRT subprocess failed rc={r.returncode}")

    def _mf6prt_parse_track_csv(self, csv_path):
        """Parse PRT track CSV into DataFrame."""
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

    # ──────────────────────────────────────────────
    # MAIN DISPATCH (called by env_base._step_run_simulation)
    # ──────────────────────────────────────────────

    def _mf6prt_reset(self):
        """Called at episode reset. Initializes persistent BMI session."""
        workdir = join(self.modelpth, 'mf6prt_persist')
        try:
            self._mf6prt_persistent_init(workdir)
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            # BMI unavailable — will fall back to one-shot per step
            self._bmi = None
            self._bmi_workdir = None

    def _mf6prt_close(self):
        """Called at episode end or before next reset."""
        self._mf6prt_persistent_close()

    def _step_run_simulation_mf6prt(self):
        """Pipeline stage 2 (MF6+PRT): run coupled flow+tracking, update trajectories."""
        from copy import copy

        # Lazy init: if BMI not yet initialized, do persistent init now
        if not hasattr(self, '_bmi') or self._bmi is None:
            workdir = join(self.modelpth, 'mf6prt_persist')
            try:
                self._mf6prt_persistent_init(workdir)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[MF6PRT] Persistent BMI init failed, falling back to one-shot: {e}")
                self._bmi = None

        if getattr(self, '_bmi', None) is not None:
            # PERSISTENT path: just advance and read
            data = self._mf6prt_persistent_step()
        else:
            # ONE-SHOT path: build + run + parse each step
            workdir = join(self.modelpth, 'mf6prt_run')
            if os.path.exists(workdir):
                shutil.rmtree(workdir)
            os.makedirs(workdir, exist_ok=True)

            sim_name = f"{self.MODELNAME}_prt"
            self._mf6prt_build_model_files(sim_name, workdir, self.periodSteps)

            try:
                data = self._mf6prt_run_bmi_oneshot(sim_name, workdir)
            except (ImportError, FileNotFoundError, RuntimeError):
                self._mf6prt_run_subprocess(sim_name, workdir)
                import glob
                track_csv = join(workdir, f'{sim_name}.trk.csv')
                if not exists(track_csv):
                    candidates = glob.glob(join(workdir, '*.trk.csv'))
                    if candidates:
                        track_csv = candidates[0]
                    else:
                        raise RuntimeError(f"Track CSV not found in {workdir}")
                data = self._mf6prt_parse_track_csv(track_csv)
            shutil.rmtree(workdir, ignore_errors=True)

        # Filter by periodLength
        valid = self._filterByTime(data)

        # Append trajectories
        for coord in 'xyz':
            series = data[coord]
            arr = np.asarray(series[valid].values) if valid is not None else np.asarray(series.values)
            self.trajectories[coord].append(arr)

        # Update particle coordinates
        coords = [float(np.asarray(data[c][valid].values)[-1]) for c in 'xyz']
        self.particleCoords = copy(coords)
