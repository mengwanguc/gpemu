/* MIT License

   Copyright (c) 2023 Gus Waldspurger

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   */

#include "alloc.h"

#include "log.h"
#include <string.h>


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>


/* Allocate shared memory using an anonymous mmap. If this process forks, and
   all "shared" state was allocated using this function, everything will behave
   properly, as if we're synchronizing threads.
   
   Returns a pointer to a SIZE-byte region of memory on success, and returns
   NULL on failure. */
void *
mmap_alloc(size_t size)
{
   /* Allocate SIZE bytes of page-aligned memory in an anonymous shared mmap. */
   assert(size > 0);
   return mmap(NULL, size,
               PROT_READ | PROT_WRITE,
               MAP_ANONYMOUS | MAP_SHARED | MAP_POPULATE,
               -1, 0);
}

/* Free memory allocated with mmap_alloc. */
void
mmap_free(void *ptr, size_t size)
{
   munmap(ptr, size);
}

/* Allocate shared mlocked memory using shm/mmap. Points PTR to a SIZE-byte
   mmapped shm object. Returns object's file descriptor on success, -errno on
   failure.
   
   Note: freeing this memory properly is non-trivial. */
int
shm_alloc(char *name, void **ptr, size_t size)
{
   /* This name shouldn't already be in use. */
   int exists = !access(name, F_OK);
   DEBUG_LOG(SCOPE_INT, LOG_DEBUG, "Opening shm object \"%s\". Already exists? %d\n", name, exists);
   assert(!exists);

   /* Create the shm object. */
   int fd = shm_open(name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
   if (fd < 0) {
      DEBUG_LOG(SCOPE_INT, LOG_ERROR, "shm_open failed; \"%s\"; %s\n", name, strerror(-fd));
      return fd;
   }

   /* Allocate SIZE bytes. */
   if (ftruncate(fd, size) < 0) {
      shm_unlink(name);
      close(fd);
      DEBUG_LOG(SCOPE_INT, LOG_ERROR, "ftruncate failed; \"%s\"; %s\n", name, strerror(errno));
      return -errno;
   }

   /* Create the mmap. */
   *ptr = mmap(NULL, size, PROT_WRITE, MAP_SHARED, fd, 0);
   if (*ptr == NULL || *ptr == MAP_FAILED) {
      shm_unlink(name);
      close(fd);
      DEBUG_LOG(SCOPE_INT, LOG_ERROR, "mmap failed; \"%s\"; %s\n", name, strerror(ENOMEM));
      return -ENOMEM;
   }

   /* Page-lock the memory. */
   if (mlock(*ptr, size) < 0) {
      shm_unlink(name);
      close(fd);
      DEBUG_LOG(SCOPE_INT, LOG_ERROR, "mlock failed; \"%s\"; %s; *ptr = %p; size = 0x%lx; *ptr + size = 0x%lx\n", name, strerror(errno), *ptr, size, ((size_t) *ptr) + size);
      return -errno;
   }

   return fd;
}

/* Free memory allocated with shm_alloc. Returns 0 on success, -errno on
   failure. */
int
shm_free(void)
{
   return -ENOSYS;
}