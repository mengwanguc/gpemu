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

#define _GNU_SOURCE

#include "minio.h"
#include "../utils/utils.h"

#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sys/shm.h>
#include <sys/mman.h>

#include "../include/uthash.h"

#define AVERAGE_FILE_SIZE (100 * 1024)
#define ENTRIES_PER_LOCK (16)
#define MIN_LOCKS (8)

#define STAT_INC(cache, field) atomic_fetch_add(&cache->field, 1)


/* Check if CACHE contains PATH. Returns true if cached, else false. */
bool
cache_contains(cache_t *c, char *path)
{
    hash_entry_t *entry = NULL;
    pthread_spin_lock(&c->ht_lock);
    HASH_FIND_STR(c->ht, path, entry);
    pthread_spin_unlock(&c->ht_lock);

    return (entry != NULL);
}

/* Store DATA into CACHE indexed by PATH. On success, returns 0. On failure,
   returns negative errno value. */
int
cache_store(cache_t *c, char *path, uint8_t *data, size_t size)
{
    /* Check size constraint. */
    if (size > c->max_item_size) {
        return -E2BIG;
    }

    /* Acquire an entry. */
    size_t n = atomic_fetch_add(&c->n_ht_entries, 1);
    if (n >= c->max_ht_entries) {
        return -ENOMEM;
    }
    hash_entry_t *entry = &c->ht_entries[n];

    /* Figure out where the data goes. */
    entry->size = size;
    size_t used = atomic_fetch_add(&c->used, size);
    entry->ptr = c->data + used;

    /* Check that this data is being placed in-range before continuing. If we're
       out-of-range, undo the expansion and abort. */
    if (used + size > c->size) {
        atomic_fetch_sub(&c->used, size);
        return -ENOMEM;
    }

    /* Copy the path into the entry. */
    strncpy(entry->path, path, MAX_PATH_LEN);

    /* Prepare the filepath according to shm requirements. */
    entry->shm_path[0] = '/';
    for (int i = 0; i < MAX_PATH_LEN + 1; i++) {
        /* Replace all occurences of '/' with '_'. */
        entry->shm_path[i + 1] = entry->path[i] == '/' ? '_' : entry->path[i];
        if (entry->path[i] == '\0') {
            break;
        }
    }

    /* Allocate an shm object for this entry's data. */
    entry->shm_fd = shm_open(entry->shm_path, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (entry->shm_fd < 0) {
        fprintf(stderr, "failed to shm_open %s\n", entry->path);
        return -errno;
    }

    /* Appropriately size the shm object. */
    if (ftruncate(entry->shm_fd, entry->size) < 0) {
        shm_unlink(entry->shm_path);
        close(entry->shm_fd);
        return -errno;
    }

    /* Create the mmap for the shm object. */
    entry->ptr = mmap(NULL, entry->size, PROT_WRITE, MAP_SHARED, entry->shm_fd, 0);
    if (entry->ptr == NULL) {
        shm_unlink(entry->shm_path);
        close(entry->shm_fd);
        return -ENOMEM;
    }

    /* Page-lock the memory. */
    mlock(entry->ptr, entry->size);

    /* Copy data to the cache. */
    memcpy(entry->ptr, data, size);

    /* Insert into hash table. */
    pthread_spin_lock(&c->ht_lock);
    HASH_ADD_STR(c->ht, path, entry);
    pthread_spin_unlock(&c->ht_lock);

    return 0;
}

/* Load the data at PATH in CACHE into DATA (a maximum of MAX bytes), storing
   the size of the file into SIZE. A cache miss is considered a failure
   (-ENODATA is returned without any IO being issued). On success returns 0.
   On failure returns negative errno. */
int
cache_load(cache_t *c, char *path, uint8_t *data, size_t *size, size_t max)
{
    hash_entry_t *entry = NULL;
    pthread_spin_lock(&c->ht_lock);
    HASH_FIND_STR(c->ht, path, entry);
    if (entry == NULL) {
        pthread_spin_unlock(&c->ht_lock);
        return -ENODATA;
    }

    pthread_spinlock_t *lock = &c->entry_locks[entry->lock_id];
    pthread_spin_lock(lock);
    pthread_spin_unlock(&c->ht_lock);

    /* Open the shm object containing the file data. Because there was a hit in
       the hashtable, an shm object with PATH must exist, and thus if this call
       fails, something is deeply broken/corrupted. */
    int fd = shm_open(entry->shm_path, O_RDWR, S_IRUSR | S_IWUSR);
    assert(fd >= 0);

    /* This call should also not fail unless something is deeply broken or
       corrupted, as this memory has already been allocated elsewhere, and given
       it's shared, there should be no real impact to system memory utilization
       from this call. */
    uint8_t *ptr = mmap(NULL, entry->size, PROT_WRITE, MAP_SHARED, fd, 0);
    assert(ptr != NULL);

    /* Copy the data into the user's DATA buffer, but don't overflow it. */
    *size = entry->size;
    if (entry->size > max) {
        pthread_spin_unlock(lock);
        return -EINVAL;
    }
    memcpy(data, ptr, entry->size);

    /* Close our references to the file data. */
    close(fd);
    munmap(ptr, entry->size);
    pthread_spin_unlock(lock);

    return 0;
}

/* Read an item from CACHE into DATA, indexed by PATH, and located on the
   filesystem at PATH. On failure returns errno code with negative value,
   otherwise returns bytes read on success.
   
   DATA must be block-aligned, in order for direct IO to work properly.
   
   Note we use atomics to implement thread safe options because pthreads and
   traditional synchronization primitives are not safe to use with the Python
   interpreter, and may cause deadlock to occur, regardless of the correctness
   of primitives' usage.
   
   It should be noted that the cache is only thread/process safe so long as the
   cache entries are only written once (as is the case with MinIO). */
ssize_t
cache_read(cache_t *c, char *path, void *data, uint64_t max_size)
{
    size_t n_accs = STAT_INC(c, n_accs);
    if (n_accs % 2500 == 0) {
        DEBUG_LOG("[MinIO debug] accesses = %lu, hits = %lu, cold misses = %lu, capacity misses = %lu, fails = %lu (usage = %lu/%lu MB) (cache->data = %p) (&cache->used = %p) (pid = %d, ppid = %d)\n", c->n_accs, c->n_hits, c->n_miss_cold, c->n_miss_capacity, c->n_fail, c->used / (1024 * 1024), c->size / (1024 * 1024), c->data, &c->used, getpid(), getppid());
    }

    /* Check if the file is cached. */
    size_t bytes = 0;
    int status = cache_load(c, path, data, &bytes, max_size);
    if (status < 0) {
        /* Don't fail if the error was the file not being cached. */
        if (status != -ENODATA) {
            return (ssize_t) status;
        }
    } else {
        STAT_INC(c, n_hits);
        return (ssize_t) bytes;
    }

    /* Open the file in DIRECT mode. */
    int fd = open(path, O_RDONLY | __O_DIRECT);
    if (fd < 0) {
        STAT_INC(c, n_fail);
        return -ENOENT;
    }

    /* Ensure the size of the file is OK. */
    size_t size = lseek(fd, 0L, SEEK_END);
    if (size > max_size || size == 0) {
        close(fd);
        STAT_INC(c, n_fail);
        return -EINVAL;
    }
    lseek(fd, 0L, SEEK_SET);

    /* Note there is an implicit assumption that two threads/processes will not
       simultaneously attempt to access the same path for the *first* time.
       For ML data-loader applications this is satisfied, since each element is
       accessed only once per epoch, however this will present a race condition
       in applications where this scenario can occur. */

    /* Read into data. */
    read(fd, data, (size | 0xFFF) + 1);
    close(fd);

    /* Cache the data. If this call fails, the data didn't fit. */
    if (cache_store(c, path, data, size) < 0) {
        STAT_INC(c, n_miss_capacity);
    } else {
        STAT_INC(c, n_miss_cold);
    }

    return size;
}

/* Clear the cache's hash table and reset used bytes to zero. */
void
cache_flush(cache_t *c)
{

    /* Free each entry's shm object. */
    hash_entry_t *entry, *tmp;
    pthread_spin_lock(&c->ht_lock);
    HASH_ITER(hh, c->ht_entries, entry, tmp) {
        pthread_spin_lock(&c->entry_locks[entry->lock_id]);
        shm_unlink(entry->shm_path);
        close(entry->shm_fd);
        munmap(entry->ptr, entry->size);
        pthread_spin_unlock(&c->entry_locks[entry->lock_id]);
    }

    /* Clear the HT and the cache metadata. */
    HASH_CLEAR(hh, c->ht);
    atomic_store(&c->used, 0);
    atomic_store(&c->n_ht_entries, 0);
    pthread_spin_unlock(&c->ht_lock);
}

/* Initialize a cache CACHE with SIZE bytes and POLICY replacement policy. On
   success, 0 is returned. On failure, negative errno value is returned. */
int
cache_init(cache_t *c,
           size_t size,
           size_t max_item_size,
           size_t avg_item_size,
           policy_t policy)
{
    /* Cache configuration. */
    c->size = size;
    c->used = 0;
    c->policy = policy;
    c->max_item_size = max_item_size;

    /* Zero initial stats. */
    c->n_accs = 0;
    c->n_fail = 0;
    c->n_hits = 0;
    c->n_miss_capacity = 0;
    c->n_miss_cold = 0;

    /* Initialize the hash table. Allocate more entries than we'll likely need,
       since file size may vary, and entries are relatively small. */
    c->n_ht_entries = 0;
    if (avg_item_size != 0) {
        c->max_ht_entries = (2 * size) / avg_item_size;
    } else {
        c->max_ht_entries = (2 * size) / AVERAGE_FILE_SIZE;
    }
    assert(c->max_ht_entries > 0);
    c->ht_size = (c->max_ht_entries + 1) * sizeof(hash_entry_t);
    if ((c->ht_entries = mmap_alloc(c->ht_size)) == NULL) {
        return -ENOMEM;
    }
    c->ht = &c->ht_entries[c->max_ht_entries];
    memset(c->ht_entries, 0, c->ht_size);

    /* Synchronization initialization. */
    assert(!pthread_spin_init(&c->ht_lock, PTHREAD_PROCESS_SHARED));
    c->n_entry_locks = MAX(MIN_LOCKS, c->max_ht_entries / ENTRIES_PER_LOCK);
    c->entry_locks = mmap_alloc(c->n_entry_locks * sizeof(pthread_spinlock_t));
    for (size_t i = 0; i < c->n_entry_locks; i++) {
        assert(!pthread_spin_init(&c->entry_locks[i], PTHREAD_PROCESS_SHARED));
    }
    for (size_t i = 0; i < c->max_ht_entries; i++) {
        c->ht_entries[i].lock_id = utils_hash(i) % c->n_entry_locks;
    }

    /* Get log2 of the number of entries. */
    int max_ht_entries_copy = c->max_ht_entries;
    int max_ht_entries_log2 = 0;
    while (max_ht_entries_copy >>= 1) max_ht_entries_log2++;
    HASH_MAKE_TABLE(hh, c->ht, 0, c->max_ht_entries, max_ht_entries_log2);

    /* We don't allocate the memory used to cache actual data yet. This memory
       will be allocated on-demand using SHM objects named with the entry's key
       in the hash table. */

    return 0;
}

/* Destroy a cache. Deallocates all allocated memory. Not thread safe. */
void
cache_destroy(cache_t *c)
{
    if (c == NULL) {
        return;
    }

    /* Free each entry's shm object. */
    hash_entry_t *entry, *tmp;
    HASH_ITER(hh, c->ht_entries, entry, tmp) {
        shm_unlink(entry->shm_path);
        close(entry->shm_fd);
        munmap(entry->ptr, entry->size);
    }

    /* Free the hash table. */
    if (c->ht_entries != NULL) {
        munmap(c->ht_entries, sizeof(hash_entry_t) * c->max_ht_entries + 1);
    }

    /* Free the spinlocks. */
    if (c->entry_locks != NULL) {
        munmap((void *) c->entry_locks, sizeof(pthread_spinlock_t) * c->n_entry_locks);
    }
}