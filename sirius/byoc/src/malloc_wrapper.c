#include <malloc_wrapper.h>
#include <stdint.h>

#ifdef PULP
#include <pulp.h>
#endif

void* malloc_wrapped(size_t size){
#ifdef PULP
    rt_alloc_t *shared_l2_allocator = &__rt_alloc_l2[2];
    void* ptr = rt_user_alloc(shared_l2_allocator, size);
#else //fallback x86 compilation
    void* ptr = malloc(size);
#endif
    return ptr;
}

void free_wrapped(void* ptr, size_t size){
#ifdef PULP
    rt_alloc_t *shared_l2_allocator = &__rt_alloc_l2[2];
    rt_user_free(shared_l2_allocator, ptr, size);
#else //fallback x86 compilation
    // This implementation actually doesn't need a wrapper, so size is ignored
    free(ptr);
#endif
}

void* malloc_wrapper(size_t size){
  // Allocate extra 4 bytes header in the first that stores the size
  size_t actual_size = size + 4;
  void* pointer = malloc_wrapped(actual_size);
  // Write to this value as if it where a uint32_t
  ((uint32_t*)pointer)[0] = (uint32_t)size;
  // return the allocated section without the header
  void* wrapped_pointer = pointer + 4;
  return wrapped_pointer;
}

void free_wrapper(void* wrapped_pointer){
  // The actual allocation started from ptr - 4
  void* actual_ptr = wrapped_pointer - 4;
  // unpack the header of the pointer
  uint32_t size = ((uint32_t*)actual_ptr)[0];
  free_wrapped(actual_ptr, size);
}
