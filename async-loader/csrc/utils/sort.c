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

#include "sort.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <alloca.h>
#include <stdbool.h>

#define SMALL_N (16)
#define MAX_STACK_BYTES (64 * 1024)

/* O(N^2) insertion sort which is fast when N is small. */
static void
sort_small(sort_wrapper_t **to_sort, size_t n)
{
    /* In each iteration, add ith element to the sorted array. We start with
       a singleton array, because a singleton array is always sorted. */
    for (uint64_t i = 1; i < n; i++) {
        sort_wrapper_t *elem = to_sort[i];

        /* Because sorted array is ascending, the first element which is ELEM
           is larger than occupies the spot where ELEM should go. */
        for (int64_t j = ((int64_t) i) - 1; j >= 0; j--) {
            if (elem->key <= to_sort[j]->key &&
                (j == 0 || elem->key > to_sort[j - 1]->key)) {
                /* Move everything to the right and insert ELEM. */
                memcpy(&to_sort[j + 1], &to_sort[j], sizeof(sort_wrapper_t *) * (i - j));
                to_sort[j] = elem;
                break;
            }
        }
    }
}

/* Merge two sorted arrays into a single sorted array, O(n). LEFT and RIGHT
   must be contigious in memory. */
static void
merge(sort_wrapper_t **left, sort_wrapper_t **right, size_t n_left, size_t n_right)
{
    size_t n = n_left + n_right;
    size_t lptr = 0, rptr = 0;

    /* Allocate an output array. If it's small, put it on the stack. If it's 
       larger, put it in the heap. */
    sort_wrapper_t **merged;
    size_t merged_size = n * sizeof(sort_wrapper_t *);
    if (merged_size > MAX_STACK_BYTES) {
        if ((merged = malloc(merged_size)) == NULL) {
            fprintf(stderr, "failed to allocate merged array\n");
            assert(false);
        }
    } else {
        merged = alloca(merged_size);
    }

    /* Walk through LEFT and RIGHT, building sorted array in MERGED. */
    for (size_t i = 0; i < n; i++) {
        /* If we've exhausted one side copy the other side in. */
        if (lptr == n_left) {
            memcpy(merged + i, right + rptr, (n - i) * sizeof(sort_wrapper_t *));
            break;
        } else if (rptr == n_right) {
            memcpy(merged + i, left + lptr, (n - i) * sizeof(sort_wrapper_t *));
            break;
        }

        /* Otherwise, advance the pointers. */
        merged[i] = left[lptr]->key <= right[rptr]->key ?
                    left[lptr++] :
                    right[rptr++];
    }

    /* Copy output into input. */
    memcpy(left, merged, n * sizeof(sort_wrapper_t *));

    /* Copy MERGED back into the input array. */
    if (merged_size > MAX_STACK_BYTES) {
        free(merged);
    }

}

/* Sort the N items in TO_SORT in ascending order. */
void
sort(sort_wrapper_t **to_sort, size_t n)
{
    /* Use insertion sort for small arrays. */
    if (n < SMALL_N) {
        sort_small(to_sort, n);
        return;
    }

    /* LHS of array. */
    size_t n_left = n / 2;
    sort_wrapper_t **left = to_sort;
    sort(left, n_left);

    /* If N is odd, include extra in RHS. */
    size_t n_right = n / 2 + n % 2;
    sort_wrapper_t **right = &to_sort[n_left];
    sort(right, n_right);

    /* Merge the two sorted arrays. */
    merge(left, right, n_left, n_right);
}