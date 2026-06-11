from keckdrpframework.primitives.base_primitive import BasePrimitive
from kcwidrp.core.bokeh_plotting import bokeh_plot
from kcwidrp.core.kcwi_plotting import save_plot
from kcwidrp.core.bokeh_plotting import bokeh_clear

from bokeh.plotting import figure
from bokeh.layouts import gridplot
import numpy as np
import math
import time
import re

def parse_fits_section(section):
    """
    Parse a FITS section string of the form '[x1:x2,y1:y2]'.

    Parameters
    ----------
    section : str
        FITS section string, e.g. '[1:1028,1:2064]'

    Returns
    -------
    tuple
        (x1, x2, y1, y2) as integers
    """
    m = re.match(r'\[(\d+):(\d+),(\d+):(\d+)\]', section.strip())
    if not m:
        raise ValueError(f"Invalid FITS section: {section}")

    return tuple(map(int, m.groups()))


class BiasDriftCorr(BasePrimitive):
    """

    Subtract redsidual bias from each amp. 

    Not well tested. Use only on short exposures with significant bias drift. 
    Use with caution. 

    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

    def _perform(self):
        # image sections for each amp
        bsec, dsec, tsec, direc, amps, aoff = self.action.args.map_ccd
        namps = len(amps)

        # perform?
        if namps == 2:
            perform = True
        else:
            perform = False

        if self.config.instrument.BIASDRIFT is None:
            perform = False

        else:
            if self.config.instrument.BIASDRIFT == False:
                perform = False
            
            if self.action.args['ttime'] > self.config.instrument.BD_MAXEXPTIME:
                perform = False

            #if 'RED' not in self.action.args.ccddata.header['CAMERA'].upper():
            #    perform = False

        if perform:

            minoscanpix = self.config.instrument.minoscanpix
            oscanbuf = self.config.instrument.oscanbuf
            frameno = self.action.args.ccddata.header['FRAMENO']
            # header keyword to update
            key = 'BDSUB'
            keycom = 'Bias Drift Corrected?'
            # is it performed?
            performed = False

            # loop over amps
            driftbiases = np.zeros(namps)
            for i, ia in enumerate(amps):
                # bias correct amp number for indexing python arrays
                iac = ia - aoff

                x0, x1, y0, y1 = parse_fits_section(self.action.args.ccddata.header['ATSEC{}'.format(ia)])

                data_slice = self.action.args.ccddata.data[y0-1:y1, x0-1:x1]
                
                bandsize = int(60 / self.action.args.xbinsize)
                if ia == 0:
                    # not sure
                    data_slice = data_slice[:, -bandsize:]
                elif ia == 1:
                    data_slice = data_slice[:, -bandsize:]
                elif ia == 2:
                    # not sure
                    data_slice = data_slice[:, :bandsize]
                elif ia == 3:
                    data_slice = data_slice[:, :bandsize]

                driftbiases[i] = np.median(data_slice)

            # matching to the first
            driftbiases = driftbiases - driftbiases[0]

            self.logger.info("Bias Drift: {}".format(driftbiases))

            for i, ia in enumerate(amps):
                # bias correct amp number for indexing python arrays
                iac = ia - aoff
                x0, x1, y0, y1 = parse_fits_section(self.action.args.ccddata.header['ATSEC{}'.format(ia)])

                self.action.args.ccddata.data[y0:y1, x0:x1] -= driftbiases[i]
                self.action.args.ccddata.header['BDVAL{}'.format(ia)] = driftbiases[i]

            performed = True

            self.action.args.ccddata.header[key] = (performed, keycom)

            log_string = BiasDriftCorr.__module__
            self.action.args.ccddata.header['HISTORY'] = log_string
            self.logger.info(log_string)

            return self.action.args
