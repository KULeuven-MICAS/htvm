# Source directories
DORY_DIR=/dory
UTILS_FILES=dory/Hardware_targets/Diana/Diana_TVM/Utils_files
DORY_HAL=dory/Hardware_targets/Diana/Backend_Kernels/dory-hal
# Destination directories
INCLUDE_DIR=dory/include/
SRC_DIR=dory/src/


mkdir -p $INCLUDE_DIR
mkdir -p $SRC_DIR

cp $DORY_DIR/$DORY_HAL/include/kernels.h $INCLUDE_DIR
cp $DORY_DIR/$DORY_HAL/include/encoders_instruction_memory.h $INCLUDE_DIR
cp $DORY_DIR/$DORY_HAL/include/utils.h $INCLUDE_DIR

cp $DORY_DIR/$DORY_HAL/src/digital_conv_2d.c $SRC_DIR
cp $DORY_DIR/$DORY_HAL/src/analog_conv_2d.c $SRC_DIR
cp $DORY_DIR/$DORY_HAL/src/digital_element_wise_sum.c $SRC_DIR
cp $DORY_DIR/$DORY_HAL/src/digital_depthwise_conv_2d.c $SRC_DIR
cp $DORY_DIR/$DORY_HAL/src/digital_fully_connected.c $SRC_DIR
cp $DORY_DIR/$DORY_HAL/src/encoders_instruction_memory.c $SRC_DIR
cp $DORY_DIR/$DORY_HAL/src/utils.c $SRC_DIR

cp $DORY_DIR/$UTILS_FILES/dory.c $SRC_DIR
cp $DORY_DIR/$UTILS_FILES/dory.h $INCLUDE_DIR
cp $DORY_DIR/$UTILS_FILES/mem_controller.c $SRC_DIR
cp $DORY_DIR/$UTILS_FILES/mem_controller.h $INCLUDE_DIR

