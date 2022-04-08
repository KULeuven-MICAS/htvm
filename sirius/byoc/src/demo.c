#include <stdio.h>
#include <stdint.h>
#include <tvm_runtime.h>
#include "tvmgen_default.h"
// Includes definition for input a and B
#include <inputs.h>
// For fancy 3D print
#include <soma_debug.h>

#define WORKSPACE_SIZE (16 * 1024)

int abs(int v) {return v * ((v > 0) - (v < 0)); }

int main(int argc, char** argv) {
	printf("Starting Demo\n");
	printf("Allocating memory\n");

	tvm_workspace_t app_workspace;
	static uint8_t g_aot_memory[WORKSPACE_SIZE];
	StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);
  int8_t *output = malloc(a_size * sizeof(int8_t));

	printf("Assigning test inputs\n");
  struct tvmgen_default_outputs outputs = {
		.output = output,
	};
	struct tvmgen_default_inputs inputs = {
		.a = a,
		.b = b,
	};

	printf("Running inference\n");
	tvmgen_default_run(&inputs, &outputs);
	printf("Inference done\n");
	debug_print_3D_tensor(0,output,"%3d",a_w, a_h, a_c);\
  printf("Program done\n");
	return 0;
}

