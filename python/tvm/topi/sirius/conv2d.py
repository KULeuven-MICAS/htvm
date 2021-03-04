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
from tvm import te  # Used to manipulate FTVMSchedules and FTVMComputes

################################### CLASSES ####################################

# Your classes go here 

################################## FUNCTIONS ###################################

def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw, specialized for SIRIUS.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    schedule: Schedule
        The computation schedule for the op.
    """
    # We start from TVM's default schedule
    schedule = te.create_schedule([x.op for x in outs])
    # Debug message
    print("Using sirius schedule")
    # Returning the built schedule
    return schedule

##################################### MAIN #####################################

if __name__ == "__main__":
    # The code to run when this file is used as a script goes here
    pass

##################################### EOF ######################################
