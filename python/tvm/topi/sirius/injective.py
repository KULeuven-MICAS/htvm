#!/usr/bin/env python3

################################### METADATA ###################################

# Contributors: Vincent Tableau Roche
# Contacts: vincent.tableau@esat.kuleuven.be
# Creation Date: 2021-03-04
# Language: Python3

################################### IMPORTS ####################################

# Standard library 
# Your imports from the standard library go here 


# External imports 
# Your imports from other packages go here 


# Internal imports 
from tvm import te  # Used for schedule manipulations
from ..utils import is_empty_shape # Used for schedule_injective

################################### CLASSES ####################################

# Your classes go here 

################################## FUNCTIONS ###################################

def schedule_injective_from_existing(sch, out):
    """Schedule for injective op from existing schedule.

    Parameters
    ----------
    sch: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the injective op.

    Returns
    -------
    sch: Schedule
         The updated schedule.
    """
    if len(sch[out].op.axis) >= 4:
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1], sch[out].op.axis[2])
        sch[out].parallel(fused)
    elif len(sch[out].op.axis) >= 3:
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1])
        sch[out].parallel(fused)
    elif len(sch[out].op.axis) >= 2:
        sch[out].parallel(sch[out].op.axis[0])
    return sch


def schedule_injective(outs):
    """SIRIUS platform schedule for injective op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of injective in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    x = outs[0]

    if list(s[x].op.axis):
        # do not vectorize for broadcast
        (io, ii) = s[x].split(list(s[x].op.axis)[-1], 4)
        s[x].vectorize(ii)
    te.schedule.AutoInlineInjective(s)

    if not is_empty_shape(x.shape):
        schedule_injective_from_existing(s, x)
    return s

##################################### MAIN #####################################

if __name__ == "__main__":
    # The code to run when this file is used as a script goes here
    pass

##################################### EOF ######################################
