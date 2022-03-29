# General settings
BUILD_DIR := build
MODEL := model

# Compiler options
PKG_COMPILE_OPTS = -g -Wall -O0 -std=gnu99

# Include paths
STANDALONE_CRT_PATH := $(abspath $(BUILD_DIR))/runtime
CODEGEN_PATH := $(abspath $(BUILD_DIR))/codegen

# Sources
CODEGEN_SRCS = $(wildcard $(abspath $(BUILD_DIR))/codegen/host/src/*.c)
CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS))

# Compiler selection
CC = gcc
AR = ar
RANLIB = ranlib

PKG_CFLAGS = ${PKG_COMPILE_OPTS}\
	     -I${STANDALONE_CRT_PATH}/include \
	     -I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
	     -I${CODEGEN_PATH}/host/include \
	     -I${PWD}/include

# These targets don't make self-titled files
.PHONY: all compile clean

all: compile $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o  $(BUILD_DIR)/libcodegen.a

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

$(BUILD_DIR)/demo: $(PWD)/src/demo.c $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o $(BUILD_DIR)/libcodegen.a
	mkdir -p $(@D)
	$(CC) $(PKG_CFLAGS) -o  $@ $^


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


