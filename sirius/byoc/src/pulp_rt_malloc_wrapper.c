#include <pulp_rt_malloc_wrapper.h>
#include <pulp.h>
#include <stdint.h>

void* malloc_pulp(size_t size){
    rt_alloc_t *shared_l2_allocator = &__rt_alloc_l2[2];
    void* ptr = rt_user_alloc(shared_l2_allocator, size);
    return ptr;
}

void* free_pulp(void* ptr, size_t size){
    rt_alloc_t *shared_l2_allocator = &__rt_alloc_l2[2];
    rt_user_free(shared_l2_allocator, ptr, size);
}

void* malloc_wrapper(size_t size){
  // Allocate extra 4 bytes header in the first that stores the size
  size_t actual_size = size + 4;
  void* pointer = malloc_pulp(actual_size);
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
  free_pulp(actual_ptr, size);
}
