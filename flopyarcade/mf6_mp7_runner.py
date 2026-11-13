#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - mf6_mp7_runner
# MODFLOW 6 + MODPATH 7 backend for all environment types.
# Mirrors the MF2005+MP6 pipeline with equivalent parameters.

import numpy as np
from os.path import join, dirname, abspath, exists
from os import makedirs
from platform import system as platformSystem
from tempfile import TemporaryDirectory
from uuid import uuid4
from numpy import ones, zeros, int32, float32, linspace, pi

from flopy.mf6 import (
    MFSimulation, ModflowTdis, ModflowIms, ModflowGwf,
    ModflowGwfdis, ModflowGwfic, ModflowGwfchd, ModflowGwfnpf,
    ModflowGwfoc, ModflowGwfrcha,
)
from flopy.mf6.modflow.mfgwfwel import ModflowGwfwel
from flopy.mf6.modflow.mfgwfsto import ModflowGwfsto
from flopy.modpath import Modpath7, Modpath7Bas


class MF6MP7Runner:
    """MODFLOW 6 + MODPATH 7 simulation runner.

    Replicates the MF2005+MP6 pipeline using MF6 packages and MP7 particle
    tracking. All parameters mirror the original build_model_other() and
    runMODPATH() methods.

    Parameters
    ----------
    self : object
        Instance containing environment configuration (from EnvBaseMixin).
    """

    def __init__(self, self_ref):
        self._s = self_ref  # reference to the environment instance

    # ------------------------------------------------------------------
    # Model construction (MF6 equivalent of build_model_other)
    # ------------------------------------------------------------------

    def build_model(self):
        """Build the full MF6 model equivalent of build_model_other()."""
        s = self._s
        makedirs(s.model_ws, exist_ok=True)

        # --- Simulation ---
        s.sim = MFSimulation(
            sim_name=s.MODELNAME,
            version='mf6',
            exe_name=s.exe_name,
            sim_ws=s.model_ws,
        )

        # --- TDIS ---
        if s.periodSteadiness:
            s.tdis = ModflowTdis(
                s.sim, time_units='DAYS', nper=1,
                perioddata=[(1.0, 1, 1.0)]
            )
        else:
            s.tdis = ModflowTdis(
                s.sim, time_units='DAYS', nper=s.periods,
                perioddata=[(2 * s.periodLength, s.periodSteps, 1.0)]
            )

        # --- IMS (matching MF2005 PCG defaults: hclose=1e-1, rclose=1e-1) ---
        s.ims = ModflowIms(
            s.sim,
            print_option='SUMMARY',
            outer_dvclose=1e-1,
            outer_maximum=100,
            inner_dvclose=1e-1,
            inner_maximum=100,
            linear_acceleration='BICGSTAB',
        )

        # --- GWF Model ---
        s.gwf = ModflowGwf(
            s.sim,
            modelname=s.MODELNAME,
            print_input=False,
            save_flows=True,
        )

        # --- DIS ---
        s.gwf_dis = ModflowGwfdis(
            s.gwf,
            nlay=s.nLay,
            nrow=s.nRow,
            ncol=s.nCol,
            delr=s.dRow,
            delc=s.dCol,
            top=s.zTop,
            botm=s.botM[1:],
        )

        # --- IC (Initial Conditions) ---
        s.gwf_ic = ModflowGwfic(s.gwf, strt=s.strt)

        # --- NPF (equivalent of LPF: hk=10, vka=10, ss=1e-5, sy=0.15) ---
        s.gwf_npf = ModflowGwfnpf(
            s.gwf,
            hk=10.0,
            vka=10.0,
            ss=1e-5,
            porosity=0.15,
        )

        # --- CHD (equivalent of boundary heads in strt) ---
        self._add_chd()

        # --- WEL ---
        self._add_wel()

        # --- RCH (for envs with recharge) ---
        if hasattr(s, 'rechTimeSeries') and s.rechTimeSeries is not None:
            rch_data = {0: s.rechTimeSeries[0]}
            s.gwf_rcha = ModflowGwfrcha(s.gwf, stress_period_data=rch_data)

        # --- OC (Output Control) ---
        s.gwf_oc = ModflowGwfoc(s.gwf, head='END', budget='END')

        # Write all input
        s.sim.write_sim()

    def _add_chd(self):
        """Add CHD boundary conditions matching _assign_boundary_heads()."""
        s = self._s
        chd_data = []
        env = s.ENVTYPE

        if env in {'1s-d', '1s-c', '1r-d', '1r-c'}:
            # West (col 0), East (col -1), South (row 0), North (row -1)
            for r in range(1, s.nRow - 1):
                chd_data.append([(0, r, 0), s.headSpecWest])
                chd_data.append([(0, r, s.nCol - 1), s.headSpecEast])
            for c in range(s.nCol):
                chd_data.append([(0, 0, c), s.actionValueSouth])
                chd_data.append([(0, s.nRow - 1, c), s.actionValueNorth])

        elif env in {'2s-d', '2s-c', '2r-d', '2r-c'}:
            for r in range(s.nRow - 1):
                chd_data.append([(0, r, 0), s.headSpecWest])
                chd_data.append([(0, r, s.nCol - 1), s.headSpecEast])
            for c in range(s.nCol):
                chd_data.append([(0, s.nRow - 1, c), s.actionValue])

        else:
            # 3s, 4s, 5s, 6s, 3r, 4r, 5r, 6r
            for r in range(1, s.nRow - 1):
                chd_data.append([(0, r, 0), s.headSpecWest])
                chd_data.append([(0, r, s.nCol - 1), s.headSpecEast])
            for c in range(s.nCol):
                chd_data.append([(0, 0, c), s.headSpecSouth])
                chd_data.append([(0, s.nRow - 1, c), s.headSpecNorth])

        s.gwf_chd = ModflowGwfchd(
            s.gwf,
            maxbound=len(chd_data),
            stress_period_data=chd_data,
        )

    def _add_wel(self):
        """Add WEL package with main and helper wells."""
        s = self._s
        l = s.wellCellLayer - 1
        c = s.wellCellColumn - 1
        r = s.wellCellRow - 1
        wells = [[l, r, c, s.wellQ]]

        if s.nHelperWells > 0:
            for i in range(s.nHelperWells):
                idx = str(i + 1)
                hl = s.helperWells[f'l{idx}'] - 1
                hc = s.helperWells[f'c{idx}'] - 1
                hr = s.helperWells[f'r{idx}'] - 1
                hq = s.helperWells[f'wellQ{idx}']
                wells.append([hl, hr, hc, hq])

        s.gwf_wel = ModflowGwfwel(
            s.gwf,
            stress_period_data={0: wells},
        )

    # ------------------------------------------------------------------
    # Run MODFLOW 6
    # ------------------------------------------------------------------

    def run_modflow(self):
        """Run the MF6 simulation and load heads."""
        s = self._s
        s.sim.run_sim(silent=True)

        # Load heads from MF6 output
        head_file = s.gwf.output().head
        s.heads = head_file.get_data()
        if hasattr(s.heads, 'to_numpy'):
            s.heads = s.heads.to_numpy()

    # ------------------------------------------------------------------
    # MODPATH 7 (equivalent of runMODPATH)
    # ------------------------------------------------------------------

    def run_modpath7(self):
        """Run MODPATH 7 particle tracking (equivalent of runMODPATH)."""
        s = self._s
        s._transform_particle_coords()

        if s.timeStep == 1:
            self._init_mp7()

        self._write_mp7_particle()
        s.mp7.run_model(silent=True)

    def _init_mp7(self):
        """Initialize MODPATH 7 objects."""
        s = self._s
        mp7_exe = join(dirname(abspath(__file__)), 'simulators', 'linux', 'mp7')
        if not exists(mp7_exe):
            raise FileNotFoundError(
                f"MP7 binary not found at {mp7_exe}. "
                "Download from https://github.com/MODPATH/modpath7/releases"
            )

        s.mp7 = Modpath7(
            s.MODELNAME,
            exe_name=mp7_exe,
            modflowmodel=s.gwf,
            model_ws=s.model_ws,
        )
        s.mp7bas = Modpath7Bas(
            s.mp7,
            prsity=0.2,
            prsityCB=0.2,
        )
        s.mp7sim = s.mp7.create_mp7sim(
            trackdir='forward',
            simtype='pathline',
            packages='RCH',
        )
        s.mp7.write_input()

    def _write_mp7_particle(self):
        """Write particle location file for MP7."""
        s = self._s
        l, c, r = s.cellInfoFromCoordinates(s.particleCoords)

        frac_col = 1.0 - ((s.particleCoords[0] / s.dCol) - int(s.particleCoords[0] / s.dCol))
        frac_row = ((s.particleCoords[1] / s.dRow) - int(s.particleCoords[1] / s.dRow))
        frac_ver = s.particleCoords[2] / s.dVer

        mploc_path = join(s.model_ws, s.MODELNAME + '.mploc')
        with open(mploc_path, 'w') as f:
            f.write('1\n1\nparticle\n1\n')
            f.write(
                f"1 1 1 {s.nLay - l + 1} {s.nRow - r + 1} "
                f"{s.nCol - c + 1} {frac_col:.6f} {frac_row:.6f} "
                f"{frac_ver:.6f} 0.000000 particle\n"
            )
            f.write('particle\n')
            f.write('1 0.000000 1\n')

    def load_particle_data(self):
        """Load particle tracking results from MP7 output."""
        s = self._s
        # MP7 outputs in .pth file (pathline) or .ptc file (travel time)
        import flopy.modpath.mp7particledata as mp7pd
        pth_file = join(s.model_ws, f'{s.MODELNAME}.pth')
        if not exists(pth_file):
            # Try .ptc
            pth_file = join(s.model_ws, f'{s.MODELNAME}.ptc')
        if not exists(pth_file):
            raise FileNotFoundError(f"MP7 output file not found in {s.model_ws}")

        import pandas as pd
        df = pd.read_csv(pth_file, header=None)
        return df
