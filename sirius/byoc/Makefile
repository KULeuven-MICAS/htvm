# General settings
BUILD_DIR := build
MODEL := model

# Compiler options
PKG_COMPILE_OPTS = -g -Wall -pedantic -O0 -std=gnu99
PULP_COMPILE_OPTS = -DSDK -DSOMA_AUTO_DEBUG -m32 -DX86_SOMA

# Include paths
STANDALONE_CRT_PATH := $(abspath $(BUILD_DIR))/runtime
CODEGEN_PATH := $(abspath $(BUILD_DIR))/codegen

# Sources
CODEGEN_SRCS = $(wildcard $(abspath $(BUILD_DIR))/codegen/host/src/*.c)
CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS))

SOMA_LIB_PATH = /home/josse/phd/sirius-high-level-library
SOMA_SRCS = $(wildcard $(SOMA_LIB_PATH)/src/*.c)
SOMA_OBJS = $(subst .c,.o,$(SOMA_SRCS))

# Compiler selection
CC = gcc
AR = ar
RANLIB = ranlib

# Compiler options and header includes
PKG_CFLAGS = ${PKG_COMPILE_OPTS}\
	     ${PULP_COMPILE_OPTS}\
	     -I${STANDALONE_CRT_PATH}/include \
	     -I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
	     -I${CODEGEN_PATH}/host/include \
	     -I${PWD}/include \
             -I${SOMA_LIB_PATH}/include

# Compiler options for SOMA

# These targets don't make self-titled files
.PHONY: all compile clean

all: compile $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o  $(BUILD_DIR)/libcodegen.a $(BUILD_DIR)/demo

$(BUILD_DIR)/stack_allocator.o: $(STANDALONE_CRT_PATH)/src/runtime/crt/memory/stack_allocator.c
	mkdir -p $(@D)
	$(CC) -c $(PKG_CFLAGS) -o $@ $^

$(BUILD_DIR)/crt_backend_api.o: $(STANDALONE_CRT_PATH)/src/runtime/crt/common/crt_backend_api.c
	mkdir -p $(@D)
	$(CC) -c $(PKG_CFLAGS) -o $@ $^

$(BUILD_DIR)/libcodegen.a: $(CODEGEN_SRCS)
	cd $(abspath $(BUILD_DIR)/codegen/host/src) && $(CC) -c $(PKG_CFLAGS) $(CODEGEN_SRCS)
	$(AR) -cr $(abspath $(BUILD_DIR)/libcodegen.a) $(CODEGEN_OBJS)
	$(RANLIB) $(abspath $(BUILD_DIR)/libcodegen.a)

$(BUILD_DIR)/demo: $(PWD)/src/demo.c $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o $(BUILD_DIR)/libcodegen.a $(SOMA_SRCS)
	mkdir -p $(@D)
	$(CC) $(PKG_CFLAGS) -o  $@ $^


compile:clean
	# Create model.tar for int8 sum
	python soma_codegen.py
	# Create build directory and move and unpack model mlf there
	mkdir -p $(BUILD_DIR)
	mv $(MODEL).tar $(BUILD_DIR)
	tar -xf $(BUILD_DIR)/$(MODEL).tar --directory=$(BUILD_DIR)
	rm $(BUILD_DIR)/$(MODEL).tar
	$(eval CODEGEN_SRCS = $(wildcard $(abspath $(BUILD_DIR))/codegen/host/src/*.c))
	$(eval CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS)))


clean:
	rm -rf $(BUILD_DIR) 


