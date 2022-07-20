#include <stdio.h>
#include <stdint.h>
#include <tvm_runtime.h>
#include "tvmgen_default.h"
// Includes definition for input a and B
// For fancy 3D print
#include <soma_debug.h>

int abs(int v) {return v * ((v > 0) - (v < 0)); }

int main(int argc, char** argv) {
	//printf("Starting Demo\n");
	//printf("Allocating memory\n");

	tvm_workspace_t app_workspace;
	static uint8_t g_aot_memory[TVMGEN_DEFAULT_WORKSPACE_SIZE];
	StackMemoryManager_Init(&app_workspace, g_aot_memory, TVMGEN_DEFAULT_WORKSPACE_SIZE);
    uint32_t input_size = 1 * 3 * 16 * 16;
    uint32_t output_size = 1 * 16 * 32 * 32;
    int8_t *input = malloc(input_size * sizeof(int8_t));
    int8_t *output = malloc(output_size * sizeof(int8_t));

	//printf("Assigning test inputs\n");
    struct tvmgen_default_outputs outputs = {
		.output = output,
	};
	struct tvmgen_default_inputs inputs = {
		.input = input,
	};

	//printf("Running inference\n");
	int32_t status = tvmgen_default_run(&inputs, &outputs);
    free(input);
    free(output);
    if(status != 0){
        abort();
    }
	//printf("Inference done\n");
	return 0;
}

