# -*- coding: utf-8 -*-
"""XFoil module to calculate aerodynamic characteristics of an airfoil.

Copyright (C) 2021 Stephan Helma
Copyright (C) 2021 GDuthe (https://github.com/GDuthe)
Copyright (C) 2019 D. de Vries
Copyright (C) 2019 DARcorporation

This file is part of xfoil-python.

xfoil-python is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

xfoil-python is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with xfoil-python.  If not, see <https://www.gnu.org/licenses/>.

"""

import ctypes
import os
import shutil
import tempfile
import pathlib

from ctypes import c_bool, c_int, c_float, byref, POINTER, cdll

import numpy as np

from .model import Airfoil

fptr = POINTER(c_float)
bptr = POINTER(c_bool)


class XFoil:
    """Interface to the XFoil Fortran routines.

    Attributes
    ----------
    airfoil
    Re
    M
    xtr
    n_crit
    max_iter
    """

    def __init__(self):
        """Initialize the `XFoil` class."""
        self._load_library()
        self._airfoil = None

    def __del__(self):
        """Delete the XFoil class instance."""
        self._unload_library()

    def _load_library(self):
        """Load the Fortran library."""
        # The libxfoil library is not threadsafe, but if we make a copy of it
        # and load that copy, all values are private to this instance!
        for lib in pathlib.Path(__file__).parent.glob('libxfoil.*'):
            # Make a temporary file
            fd, tmp_lib = tempfile.mkstemp(prefix=lib.stem, suffix=lib.suffix)
            # Close that file, so that Windows will not complain
            os.close(fd)
            # Copy the library to the temporary file
            shutil.copy2(lib, tmp_lib)
            # Load the library
            try:
                self._lib = cdll.LoadLibrary(tmp_lib)
            except OSError:
                os.remove(tmp_lib)
                continue
            except Exception:
                os.remove(tmp_lib)
                raise
            else:
                self._lib_path = tmp_lib
                break
        else:
            raise RuntimeError("Could not load the runtime library 'libxfoil.*'")

        self._lib.init()

        self._lib.get_print.restype = c_bool
        self._lib.get_reynolds.restype = c_float
        self._lib.get_mach.restype = c_float
        self._lib.get_n_crit.restype = c_float

    def _unload_library(self):
        """Unload the Fortran library and delete its temporary copy."""
        handle = self._lib._handle
        del self._lib
        try:
            ctypes.windll.kernel32.FreeLibrary(handle)
        except AttributeError:
            pass
        finally:
            os.remove(self._lib_path)

    @property
    def print(self):
        """bool: True if console output should be shown."""
        return self._lib.get_print()

    @print.setter
    def print(self, value):
        self._lib.set_print(byref(c_bool(value)))

    @property
    def airfoil(self):
        """Airfoil: Instance of the Airfoil class."""
        n = self._lib.get_n_coords()
        x = np.asfortranarray(np.zeros(n), dtype=c_float)
        y = np.asfortranarray(np.zeros(n), dtype=c_float)
        self._lib.get_airfoil(
            x.ctypes.data_as(fptr), y.ctypes.data_as(fptr), byref(c_int(n)))
        return Airfoil(x.astype(float), y.astype(float))

    @airfoil.setter
    def airfoil(self, airfoil):
        self._airfoil = airfoil
        self._lib.set_airfoil(
            np.asfortranarray(
                airfoil.x.flatten(), dtype=c_float).ctypes.data_as(fptr),
            np.asfortranarray(
                airfoil.y.flatten(), dtype=c_float).ctypes.data_as(fptr),
            byref(
                c_int(airfoil.n_coords))
        )

    @property
    def Re(self):
        """float: Reynolds number."""
        return float(self._lib.get_reynolds())

    @Re.setter
    def Re(self, value):
        self._lib.set_reynolds(byref(c_float(value)))

    @property
    def M(self):
        """float: Mach number."""
        return float(self._lib.get_mach())

    @M.setter
    def M(self, value):
        self._lib.set_mach(byref(c_float(value)))

    @property
    def xtr(self):
        """tuple(float, float): Top and bottom flow trip x/c locations."""
        xtr_top = c_float()
        xtr_bot = c_float()
        self._lib.get_xtr(byref(xtr_top), byref(xtr_bot))
        return float(xtr_top), float(xtr_bot)

    @xtr.setter
    def xtr(self, value):
        self._lib.set_xtr(byref(c_float(value[0])), byref(c_float(value[1])))

    @property
    def n_crit(self):
        """float: Critical amplification ratio."""
        return float(self._lib.get_n_crit())

    @n_crit.setter
    def n_crit(self, value):
        self._lib.set_n_crit(byref(c_float(value)))

    @property
    def max_iter(self):
        """int: Maximum number of iterations."""
        return int(self._lib.get_max_iter())

    @max_iter.setter
    def max_iter(self, max_iter):
        self._lib.set_max_iter(byref(c_int(max_iter)))

    def naca(self, specifier):
        """Set a NACA 4 or 5 series airfoil.

        Parameters
        ----------
        specifier : string
            A NACA 4 or 5 series identifier, such as '2412'.
        """
        self._lib.set_naca(byref(c_int(int(specifier))))

    def reset_bls(self):
        """Reset boundary layers to be reinitialized on the next analysis."""
        self._lib.reset_bls()

    def repanel(self,
                n_nodes=160, cv_par=1, cte_ratio=0.15, ctr_ratio=0.2,
                xt_ref=(1, 1), xb_ref=(1, 1)):
        """Re-panel airfoil.

        Parameters
        ----------
        n_nodes : int
            Number of panel nodes
        cv_par : float
            Panel bunching parameter
        cte_ratio : float
            TE/LE panel density ratio
        ctr_ratio : float
            Refined-area/LE panel density ratio
        xt_ref : tuple of two floats
            Top side refined area x/c limits
        xb_ref : tuple of two floats
            Bottom side refined area x/c limits
        """
        self._lib.repanel(byref(c_int(n_nodes)), byref(c_float(cv_par)),
                          byref(c_float(cte_ratio)), byref(c_float(ctr_ratio)),
                          byref(c_float(xt_ref[0])), byref(c_float(xt_ref[1])),
                          byref(c_float(xb_ref[0])), byref(c_float(xb_ref[1])))

    def filter(self, factor=0.2):
        """Filter surface speed distribution using modified Hanning filter.

        Parameters
        ----------
        factor : float
            Filter parameter. If set to 1, the standard, full Hanning filter
            is applied.
            Default is 0.2.
        """
        self._lib.filter(byref(c_float(factor)))

    def a(self, a):
        """Analyze airfoil at a fixed angle of attack.

        Parameters
        ----------
        a : float
            Angle of attack in degrees

        Returns
        -------
        cl, cd, cm, cp : float
            Corresponding values of the lift, drag, moment, and minimum
            pressure coefficients.
        """
        cl = c_float()
        cd = c_float()
        cm = c_float()
        cp = c_float()
        conv = c_bool()

        self._lib.alfa(
            byref(c_float(a)),
            byref(cl), byref(cd), byref(cm), byref(cp),
            byref(conv))

        if conv:
            return cl.value, cd.value, cm.value, cp.value
        else:
            return np.nan, np.nan, np.nan, np.nan

    def cl(self, cl):
        """Analyze airfoil at a fixed lift coefficient.

        Parameters
        ----------
        cl : float
            Lift coefficient

        Returns
        -------
        a, cd, cm, cp : float
            Corresponding values of the angle of attack, drag, moment, and
            minimum pressure coefficients.
        """
        a = c_float()
        cd = c_float()
        cm = c_float()
        cp = c_float()
        conv = c_bool()

        self._lib.cl(
            byref(c_float(cl)),
            byref(a), byref(cd), byref(cm), byref(cp),
            byref(conv))

        if conv:
            return a.value, cd.value, cm.value, cp.value
        else:
            return np.nan, np.nan, np.nan, np.nan

    def aseq(self, a_start, a_end, a_step):
        """Analyze airfoil at a sequence of angles of attack.

        The analysis is done for the angles of attack given by
        range(a_start, a_end, a_step).

        Parameters
        ----------
        a_start, a_end, a_step : float
            Start, end, and increment angles for the range.

        Returns
        -------
        a, cl, cd, cm, co : np.ndarray
            Lists of angles of attack and their corresponding lift, drag,
            moment, and minimum pressure coefficients.
        """
        n = abs(int((a_end - a_start) / a_step))

        a = np.zeros(n, dtype=c_float)
        cl = np.zeros(n, dtype=c_float)
        cd = np.zeros(n, dtype=c_float)
        cm = np.zeros(n, dtype=c_float)
        cp = np.zeros(n, dtype=c_float)
        conv = np.zeros(n, dtype=c_bool)
        arr1 = np.zeros(n, dtype=c_float)
        arr2 = np.zeros(n, dtype=c_float)
        arr3 = np.zeros(n, dtype=c_float)
        arr4 = np.zeros(n, dtype=c_float)
        arr5 = np.zeros(n, dtype=c_float)
        arr6 = np.zeros(n, dtype=c_float)
        self._lib.aseq(
            byref(c_float(a_start)), byref(c_float(a_end)), byref(c_int(n)),
            a.ctypes.data_as(fptr), cl.ctypes.data_as(fptr),
            cd.ctypes.data_as(fptr), cm.ctypes.data_as(fptr),
            cp.ctypes.data_as(fptr), conv.ctypes.data_as(bptr),
            arr1.ctypes.data_as(fptr),
            arr2.ctypes.data_as(fptr),
            arr3.ctypes.data_as(fptr),
            arr4.ctypes.data_as(fptr),
            arr5.ctypes.data_as(fptr),
            arr6.ctypes.data_as(fptr),
            )

        isnan = np.logical_not(conv)
        a[isnan] = np.nan
        cl[isnan] = np.nan
        cd[isnan] = np.nan
        cm[isnan] = np.nan
        cp[isnan] = np.nan
        arr1[isnan] = np.nan
        arr2[isnan] = np.nan
        arr3[isnan] = np.nan
        arr4[isnan] = np.nan
        arr5[isnan] = np.nan
        arr6[isnan] = np.nan

        return (
            a.astype(float),
            cl.astype(float), cd.astype(float),
            cm.astype(float), cp.astype(float),
            conv.astype(float), 
            arr1.astype(float), arr2.astype(float), arr3.astype(float), arr4.astype(float), arr5.astype(float), arr6.astype(float)
            )

    def cseq(self, cl_start, cl_end, cl_step):
        """Analyze airfoil at a sequence of lift coefficients.

        The analysis is done for the lift coefficients given by
        range(cl_start, cl_end, cl_step).

        Parameters
        ----------
        cl_start, cl_end, cl_step : float
            Start, end, and increment lift coefficients for the range.

        Returns
        -------
        a, cl, cd, cm, co : np.ndarray
            Lists of angles of attack and their corresponding lift, drag,
            moment, and minimum pressure coefficients.
        """
        n = abs(int((cl_end - cl_start) / cl_step))

        a = np.zeros(n, dtype=c_float)
        cl = np.zeros(n, dtype=c_float)
        cd = np.zeros(n, dtype=c_float)
        cm = np.zeros(n, dtype=c_float)
        cp = np.zeros(n, dtype=c_float)
        conv = np.zeros(n, dtype=c_bool)

        self._lib.cseq(
            byref(c_float(cl_start)), byref(c_float(cl_end)), byref(c_int(n)),
            a.ctypes.data_as(fptr), cl.ctypes.data_as(fptr),
            cd.ctypes.data_as(fptr), cm.ctypes.data_as(fptr),
            cp.ctypes.data_as(fptr), conv.ctypes.data_as(bptr))

        isnan = np.logical_not(conv)
        a[isnan] = np.nan
        cl[isnan] = np.nan
        cd[isnan] = np.nan
        cm[isnan] = np.nan
        cp[isnan] = np.nan

        return (
            a.astype(float),
            cl.astype(float), cd.astype(float),
            cm.astype(float), cp.astype(float))

    def get_cp_distribution(self):
        """Get the Cp distribution from the last converged point.

        Returns
        -------
        x : np.array
            X-coordinates
        y : np.array
            Y-coordinates
        cp : np.ndarray
            Pressure coefficients at the corresponding x-coordinates
        """
        n = self._lib.get_n_cp()
        x = np.zeros(n, dtype=c_float)
        y = np.zeros(n, dtype=c_float)
        cp = np.zeros(n, dtype=c_float)

        self._lib.get_cp(
            x.ctypes.data_as(fptr), y.ctypes.data_as(fptr),
            cp.ctypes.data_as(fptr), byref(c_int(n)))

        return x.astype(float), y.astype(float), cp.astype(float)
