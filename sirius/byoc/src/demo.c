#include <stdio.h>
#include <stdint.h>
#include <tvm_runtime.h>
#include "tvmgen_default.h"

#define WORKSPACE_SIZE (16 * 1024)

int abs(int v) {return v * ((v > 0) - (v < 0)); }

int main(int argc, char** argv) {
	printf("Starting Demo\n");

	printf("Allocating memory\n");
	tvm_workspace_t app_workspace;
	static uint8_t g_aot_memory[WORKSPACE_SIZE];
	StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);

	printf("Creating test inputs\n");
	uint32_t w = 3;
	uint32_t h = 3;
	uint32_t c = 3;
	int8_t *a = malloc(w * h * c * sizeof(int8_t));
	int8_t *b = malloc(w * h * c * sizeof(int8_t));
	int8_t *output = malloc(w * h * c * sizeof(int8_t));
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
		    for (int k = 0; k < c; k++) {
			// Getting the position in the row-major-ordered arrays.
			unsigned int position = i * h * c + j * c + k;
			// One value for A.
			a[position] = i * j + k;
			// One value for B.
			b[position] = i + j - 2 * k;
		    }
		}
	}
	printf("Running inference\n");
	struct tvmgen_default_outputs outputs = {
		.output = output,
	};
	struct tvmgen_default_inputs inputs = {
		.a = a,
		.b = b,
	};
	tvmgen_default_run(&inputs, &outputs);
	printf("Inference done\n");
	return 0;
}

