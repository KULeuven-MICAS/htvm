 
#include <stdio.h>
#include <stdint.h>
#include "tvmgen_default.h"
#include <tvm_runtime.h>
    
int abs(int v) {return v * ((v > 0) - (v < 0)); }

int main(int argc, char** argv) {
    tvm_workspace_t app_workspace;
    static uint8_t g_aot_memory[TVMGEN_DEFAULT_WORKSPACE_SIZE];
    StackMemoryManager_Init(&app_workspace, g_aot_memory, TVMGEN_DEFAULT_WORKSPACE_SIZE);
    // Sizes automatically added by utils.create_demo_file
    	uint32_t input_size = 768;
	uint32_t output_size = 8192;

        int8_t *input = malloc(input_size * sizeof(int8_t));
        int8_t *output = malloc(output_size * sizeof(int8_t));
        
    // Fill first input with ones
    for (uint32_t i = 0; i < input_size; i++){
        input[i] = 1;
    }

    struct tvmgen_default_outputs outputs = {
    	.output = output,
    };
    struct tvmgen_default_inputs inputs = {
    	.input = input,
    };
    int32_t status = tvmgen_default_run(&inputs, &outputs);
    
        free(input);
        free(output);
        
    if(status != 0){
        abort();
    }
    return 0;
}
    