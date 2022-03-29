# General settings
BUILD_DIR := build
MODEL := model

# Compiler options
PKG_COMPILE_OPTS = -g -Wall -O0 -std=gnu99

# Include paths
STANDALONE_CRT_PATH := $(abspath $(BUILD_DIR))/runtime
CODEGEN_PATH := $(abspath $(BUILD_DIR))/codegen

# Sources
CODEGEN_SRCS = $(wildcard $(CODEGEN_PATH)/host/src/*.c)
CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS))

# Compiler selection
CC = gcc
AR = ar
RANLIB = ranlib

PKG_CFLAGS = ${PKG_COMPILE_OPTS}\
	     -I${STANDALONE_CRT_PATH}/include \
	     -I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
	     -I${CODEGEN_PATH}/include \

$(ifeq VERBOSE,1)
QUIET ?=
$(else)
QUIET ?= @
$(endif)

all: compile

compile:
	# Create model.tar for int8 sum
	python soma_codegen.py
	# Create build directory and move and unpack model mlf there
	mkdir -p $(BUILD_DIR)
	mv $(MODEL).tar $(BUILD_DIR)
	tar -xf $(BUILD_DIR)/$(MODEL).tar --directory=$(BUILD_DIR)
	rm $(BUILD_DIR)/$(MODEL).tar

clean:
	rm -rf $(BUILD_DIR) 


