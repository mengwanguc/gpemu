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

#include "../../../csrc/utils/sort.h"
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

#define N_KEYS (35)

int
main(int argc, char **argv)
{
    printf("Testing sorting...");

    uint64_t keys_random[N_KEYS] = {
        26, 35, 86, 52, 59, 95, 46, 97, 60, 83, 63, 56, 57, 30, 63, 26, 92, 94,
        69, 37, 66, 49, 95, 7, 38, 53, 36, 73, 22, 73, 7, 99, 21, 64, 66
    };
    uint64_t keys_sorted[N_KEYS] = {
        7, 7, 21, 22, 26, 26, 30, 35, 36, 37, 38, 46, 49, 52, 53, 56, 57, 59,
        60, 63, 63, 64, 66, 66, 69, 73, 73, 83, 86, 92, 94, 95, 95, 97, 99
    };

    /* Set up wrappers. */
    sort_wrapper_t wrappers[N_KEYS];
    for (size_t i = 0; i < N_KEYS; i++) {
        wrappers[i].key = keys_random[i];
    }

    /* Set up array of pointers. */
    sort_wrapper_t *ptrs[N_KEYS];
    for (size_t i = 0; i < N_KEYS; i++) {
        ptrs[i] = &wrappers[i];
    }

    /* Sort. */
    sort(ptrs, N_KEYS);

    /* Verify results. */
    for (size_t i = 0; i < N_KEYS; i++) {
        if (ptrs[i]->key != keys_sorted[i]) {
            printf("failed; %lu != %lu\n", ptrs[i]->key, keys_sorted[i]);
            return EXIT_FAILURE;
        }
    }

    printf("success\n");
    return EXIT_SUCCESS;
}