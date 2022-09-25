# Source directory
DORY_DIR=/dory
# Destination directories
INCLUDE_DIR=dory/include/
SRC_DIR=dory/src/

mkdir -p $INCLUDE_DIR
mkdir -p $SRC_DIR

cp $DORY_DIR/dory/Hardware_targets/Diana/Backend_Kernels/dory-hal/include/digital_conv_2d.h $INCLUDE_DIR
#cp $DORY_DIR/dory/Hardware_targets/Diana/Backend_Kernels/dory-hal/include/digital_element_wise_sum.h $INCLUDE_DIR
cp $DORY_DIR/dory/Hardware_targets/Diana/Backend_Kernels/dory-hal/include/digital_encoder_instruction_memory.h $INCLUDE_DIR
cp $DORY_DIR/dory/Hardware_targets/Diana/Backend_Kernels/dory-hal/include/utils.h $INCLUDE_DIR

cp $DORY_DIR/dory/Hardware_targets/Diana/Backend_Kernels/dory-hal/src/digital_conv_2d.c $SRC_DIR
#cp $DORY_DIR/dory/Hardware_targets/Diana/Backend_Kernels/dory-hal/src/digital_element_wise_sum.c $SRC_DIR
cp $DORY_DIR/dory/Hardware_targets/Diana/Backend_Kernels/dory-hal/src/digital_encoder_instruction_memory.c $SRC_DIR
cp $DORY_DIR/dory/Hardware_targets/Diana/Backend_Kernels/dory-hal/src/utils.c $SRC_DIR

cp $DORY_DIR/dory/Hardware_targets/Diana/Diana_TVM/Utils_files/dory.c $SRC_DIR
cp $DORY_DIR/dory/Hardware_targets/Diana/Diana_TVM/Utils_files/dory.h $INCLUDE_DIR
cp $DORY_DIR/dory/Hardware_targets/Diana/Diana_TVM/Utils_files/mem_controller.c $SRC_DIR
cp $DORY_DIR/dory/Hardware_targets/Diana/Diana_TVM/Utils_files/mem_controller.h $INCLUDE_DIR

