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

#include <stdlib.h>
#include <assert.h>
#include <sys/mman.h>

#define _GNU_SOURCE


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
