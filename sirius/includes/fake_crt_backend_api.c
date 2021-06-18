/* Alternative to tvm's crt_backend_api.c */
#include <stdlib.h>
#include <stdint.h>

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
  /*
  tvm_crt_error_t err = kTvmErrorNoError;
  void* ptr = 0;
  DLContext ctx = {device_type, device_id};
  assert(nbytes > 0);
  err = TVMPlatformMemoryAllocate(nbytes, ctx, &ptr);
  CHECK_EQ(err, kTvmErrorNoError,
           "TVMBackendAllocWorkspace(%d, %d, %" PRIu64 ", %d, %d) -> %" PRId32, device_type,
      device_id, nbytes, dtype_code_hint, dtype_bits_hint, err);
  return ptr;*/

  // Just like malloc this function returns the pointer to the allocated memory space.
  // If a NULL pointer is returned the allocation did not happen properly
  // We disregard the use of device_type, device_id, dtype_code_hint, and dtype_bits_hint fields
  void *ptr = malloc(nbytes * sizeof(int8));
  // malloc will return a null pointer if allocation fails, just like TVMBackendAllocWorkspace
  return ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  /*
  tvm_crt_error_t err = kTvmErrorNoError;
  DLContext ctx = {device_type, device_id};
  err = TVMPlatformMemoryFree(ptr, ctx);
  return err;*/
  // We disregard device_type and device_id in this code
  free(ptr);
  // function returns zero when free has happened successfully
  // Free does not return any value, so we just always return 0
  return(0);
}
