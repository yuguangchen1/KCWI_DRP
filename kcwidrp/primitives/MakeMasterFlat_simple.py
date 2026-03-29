from keckdrpframework.primitives.base_img import BaseImg
from kcwidrp.primitives.kcwi_file_primitives import kcwi_fits_reader, \
    kcwi_fits_writer, strip_fname, plotlabel
from kcwidrp.core.kcwi_plotting import get_plot_lims
from kcwidrp.core.bokeh_plotting import bokeh_plot
from kcwidrp.core.kcwi_plotting import save_plot
from kcwidrp.core.bspline import Bspline
from bokeh.plotting import figure
from bokeh.models import Range1d

import os
import time
import numpy as np
from scipy.signal.windows import boxcar
import scipy as sp
from scipy.signal import find_peaks


def bm_ledge_position(cwave, dich):
    if dich:
        fit = [0.240742, 4060.56]
    else:
        fit = [0.240742, 4044.56]
    return fit[1] + fit[0] * cwave


class MakeMasterFlat(BaseImg):
    """
    Generate illumination correction from a stacked flat image.

    Uses b-spline fits along with geometry maps to generate a master image for
    illumination correction.  If the flat is internal, accounts for vignetting
    along one edge.  Also accounts for ledge seen in BM grating.

    Depending on the type of input stack, the following files are written out
    and entries are made in the proc file:

        * SFLAT - a \*_mflat.fits file and an MFLAT entry
        * SDOME - a \*_mdome.fits file and an MDOME entry
        * STWIF - a \*_mtwif.fits file and an MTWIF entry

"""

    def __init__(self, action, context):
        BaseImg.__init__(self, action, context)
        self.logger = context.pipeline_logger

    def _pre_condition(self):
        """
        Checks if we can create a master flat based on the processing table
        """
        # check for flat stack to use in generating master flat
        self.logger.info("Checking precondition for MakeMasterFlat")
        tab = self.context.proctab.search_proctab(
            frame=self.action.args.ccddata,
            target_type=self.action.args.new_type, nearest=True)
        if len(tab) > 0:
            self.logger.info("already have %d master %s flats, "
                             " expecting 0" % (len(tab),
                                               self.action.args.new_type))
            return False
        else:
            self.logger.info("No %s master flat found, "
                             "checking for flat stack." %
                             self.action.args.new_type)
            self.stack_list = self.context.proctab.search_proctab(
                frame=self.action.args.ccddata,
                target_type=self.action.args.stack_type,
                target_group=self.action.args.groupid)
            self.logger.info(f"pre condition got {len(self.stack_list)},"
                             f" expecting {self.action.args.min_files}")
            # do we meet the criterion?
            if len(self.stack_list) >= 1:
                return True
            else:
                return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation
        """
        self.logger.info("Creating master illumination correction")
        olab = plotlabel(self.action.args)

        suffix = self.action.args.new_type.lower()
        insuff = self.action.args.stack_type.lower()

        stack_list = list(self.stack_list['filename'])

        if len(stack_list) <= 0:
            self.logger.warning("No flat stack found!")
            return self.action.args

        # get root for maps
        tab = self.context.proctab.search_proctab(
            frame=self.action.args.ccddata, target_type='MARC',
            target_group=self.action.args.groupid)
        if len(tab) <= 0:
            self.logger.error("Geometry not solved!")
            return self.action.args

        mroot = strip_fname(tab['filename'][-1]) # ??????????????????

        # Read in stacked flat image
        stname = strip_fname(stack_list[0]) + '_' + insuff + '.fits'

        self.logger.info("Reading image: %s" % stname)
        stacked = kcwi_fits_reader(os.path.join(
            self.config.instrument.cwd, 'redux', stname))[0]
        plab = " ".join(olab.split()[0:3]) + (
            " %d " % stacked.header['FRAMENO']
        ) + " ".join(olab.split()[4:])

        # get type of flat
        internal = ('SFLAT' in stacked.header['IMTYPE'])
        twiflat = ('STWIF' in stacked.header['IMTYPE'])
        domeflat = ('SDOME' in stacked.header['IMTYPE'])

        if internal:
            self.logger.info("Internal Flat")
        elif twiflat:
            self.logger.info("Twilight Flat")
        elif domeflat:
            self.logger.info("Dome Flat")
        else:
            self.logger.error("Flat of Unknown Type!")
            return self.action.args

        mfname = strip_fname(stack_list[0]) + '_' + suffix + '.fits'

        # store flat in output frame
        stacked.data = 1/(0.001*stacked.data.copy()) #ratio
        # stacked.data[wavemap.data < 0] = 0

        # output master flat
        kcwi_fits_writer(stacked, output_file=mfname,
                         output_dir=self.config.instrument.output_directory)
        self.context.proctab.update_proctab(frame=stacked, suffix=suffix,
                                            newtype=self.action.args.new_type,
                                            filename=stack_list[0]) ### HERE
        self.context.proctab.write_proctab(tfil=self.config.instrument.procfile)
        # self.action.args.name = stacked.header['OFNAME']
        # self.action.args.name = mfname

        self.logger.info(log_string)
        return self.action.args

    # END: class MakeMasterFlat()
