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

#ifndef __MINIO_MODULE_H_
#define __MINIO_MODULE_H_

#include <stdlib.h>

#include "../include/uthash.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/mman.h>

#define MAX_PATH_LEN 128

/* Cache replacement policy. */
typedef enum {
    POLICY_FIFO,
    POLICY_MINIO,
    N_POLICIES
} policy_t;

/* Hash table entry. Maps filepath to cached data. An entry must be in the hash
   table IFF the corresponding file is cached. */
typedef struct {
    char      path[MAX_PATH_LEN + 1];       /* Key. Filepath of file. */
    char      shm_path[MAX_PATH_LEN + 2];   /* PATH but with '/' replaced with
                                               '_' to name the shm object. */
    void     *ptr;                          /* Pointer to this file's data. */
    size_t    size;                         /* Size of file data in bytes. */
    int       shm_fd;                       /* SHM object file descriptor. */
    uint64_t  lock_id;                      /* ID of lock in ENTRY_LOCKS array
                                               for this entry's protection. */

    UT_hash_handle hh;
} hash_entry_t;

/* Cache. Atomics types are used to ensure thread safety. */
typedef struct {
    /* Configuration. */
    policy_t policy;            /* Replacement policy. Only MinIO supported. */
    size_t   size;              /* Size of cache in bytes. */
    size_t   ht_size;           /* Number of bytes allocated for HT entries. */
    size_t   max_ht_entries;    /* Maximum number of HT entries. */
    size_t   max_item_size;     /* Maximum size of an element in the cache. All
                                   reads for larger items bypass the cache. A
                                   size of zero indicates there is no limit. */

    /* State. */
    atomic_size_t  used;            /* Number of bytes cached. */
    uint8_t       *data;            /* First byte of SIZE bytes of memory. */
    hash_entry_t  *ht_entries;      /* Memory used for HT entries. */
    atomic_size_t  n_ht_entries;    /* Current number of HT entries. */
    hash_entry_t  *ht;              /* Hash table, maps filename to data. */

    /* Statistics. */
    atomic_size_t n_accs;
    atomic_size_t n_hits;
    atomic_size_t n_miss_cold;
    atomic_size_t n_miss_capacity;
    atomic_size_t n_fail;

    /* Synchronization. */
    pthread_spinlock_t  ht_lock;        /* Protects hash table. */
    pthread_spinlock_t *entry_locks;    /* Protects hash table entries. Indexed
                                           using a hash function to provide fair
                                           coverage. Size of the array decides
                                           maximum parallelism. */
    size_t              n_entry_locks;  /* Number of locks in ENTRY_LOCKS. */
} cache_t;

bool cache_contains(cache_t *cache, char *path);
int cache_store(cache_t *cache, char *path, uint8_t *data, size_t size);
int cache_load(cache_t *cache, char *path, uint8_t *data, size_t *size, size_t max);
ssize_t cache_read(cache_t *cache, char *filepath, void *data, uint64_t max_size);
void cache_flush(cache_t *cache);
int cache_init(cache_t *cache, size_t size, size_t max_item_size, size_t avg_item_size, policy_t policy);
void cache_destroy(cache_t *c);

#endif