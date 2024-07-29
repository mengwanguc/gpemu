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

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <liburing.h>
#include <netinet/in.h>
#include "../utils/uthash.h"

#ifndef __CACHE_H__
#define __CACHE_H__

#define NOT_REACHED() assert(false)

#define MAX_IDLE_ITERS 64 * 1024 * 1024     /* Max idle iters before syncing. */
#define MAX_PATH_LEN 128                    /* Not including \0. */
#define MAX_SHM_PATH_LEN MAX_PATH_LEN + 1   /* Not including \0. */
#define MAX_NAME_LEN 128                    /* Not including \0. */
#define MAX_SYNC_SIZE 16 * 1024 * (MAX_PATH_LEN + 1)

#define HEADER_MAGIC (0xADDA)

#define FLAG_NONE (0b00000000)
#define FLAG_CRCT (0b00001000)  /* This message is a correction. */
#define FLAG_UNBL (0b00000100)  /* The request was unable to be fulfilled. */
#define FLAG_RPLY (0b00000010)  /* Unused flag. */
#define FLAG_FLG4 (0b00000001)  /* Unused flag. */

/* Network message type. Number of types must not exceed 16. */
typedef enum {
    TYPE_RQST,  /* File transfer request. Reply expected (TYPE_RSPN). */
    TYPE_RSPN,  /* File transfer response. No reply. */
    TYPE_SYNC,  /* File ownership synchronization. No reply. */
    TYPE_HLLO,  /* Registration (hello) message. Reply optional (TYPE_HLLO). */
    N_MTYPES
} mtype_t;

/* File load request (queue entry). Shared memory accessible by both API users
   and the ladcache loader process. */
typedef struct file_request {
    /* Request status. */
    int status;   /* Status of request, 0 on success, -errno on failure. */

    /* File metadata. */
    size_t  size;                           /* Size of file data in bytes. */
    size_t  shm_size;                       /* Size of shm object in bytes. */
    char    path[MAX_PATH_LEN + 1];         /* Path to file, +1 for \0.  */
    char    shm_path[MAX_SHM_PATH_LEN + 1]; /* PATH, but for shm object, and
                                               conforming to shm name rules,
                                               +1 for \0. */

    /* Loader state. */
    void *_ldata;       /* Loader's pointer to the shm object's memory. */
    int   _lfd_shm;     /* Loader's FD for the shm object. */
    int   _lfd_file;    /* Loader's FD for the file being loaded. */
    bool  _skip_clean;  /* Whether to skip shm cleanup for this request. Set to
                           prevent shm cleanup for newly cached items. */

    /* User state. */
    void *udata;        /* User's pointer to the shm object's memory. */
    int   ufd_shm;      /* User's FD for the shm object. */

    /* Free list. */
    struct file_request *next;
    struct file_request *prev;

    UT_hash_handle hh;
} request_t;

/* File data location for the local cache. Manager/monitor context. */
typedef struct local_location {
    char    path[MAX_PATH_LEN + 1];             /* Hash table key, +1 for \0. */
    char    shm_path[MAX_SHM_PATH_LEN + 1];     /* Shm obj. path, +1 for \0. */
    int     shm_fd;                             /* FD for shm object. */
    size_t  shm_size;                           /* Shm object size in bytes. */
    size_t  size;                               /* Size of file in bytes. */
    void   *data;                               /* Shm object's memory. */

    /* Update (new) list. */
    struct local_location *next;    /* Next entry in the new list. */
    struct local_location *prev;    /* Previous entry in the new list. */

    UT_hash_handle hh;
} lloc_t;

/* Local cache state. */
typedef struct {
    lloc_t *ht;           /* Hash table. */
    lloc_t *unsynced;     /* Link list of unsynced filenames. */
    size_t  n_unsynced;   /* Length of UNSYNCED. */
    size_t  threshold;    /* Maximum length of UNSYNCED before the list is
                             flushed to all known peers. */
    size_t  capacity;     /* Maximum capacity of cache in bytes. */
    size_t  used;         /* Current usage of cache in bytes. */
} lcache_t;

/* General purpose network message struct. Messages follow the format below.

        Hello    ┌────────┐
        Message  │header  │ (UDP only)
        (0011)   │7 bytes │
        ──────►  └────────┘

        Sync     ┌────────┬─────────┬────────┬────────┬───┬────────┐
        Message  │header  │# entries│filepath│filepath│   │filepath│ (TCP only)
        (0010)   │7 bytes │4 bytes  │n bytes │n bytes │...│n bytes │
        ──────►  └────────┴─────────┴────────┴────────┴───┴────────┘

        File     ┌────────┬─────────┐
        Response │header  │file data│ (TCP only)
        (0001)   │7 bytes │n bytes  │
        ──────►  └────────┴─────────┘

        File     ┌────────┬────────┐
        Request  │header  │filepath│ (TCP only)
        (0000)   │7 bytes │n bytes │
        ──────►  └────────┴────────┘

                        ┌───────┬──────┬──┬──┬──┬──┬───────┬───────┐
                header  │0xADDA │type  │CR│UN│RP│FL│length │random │
                format  │2 bytes│4 bits│CT│BL│LY│G3│4 bytes│4 bytes│
                        └───────┴──────┴──┴──┴──┴──┴───────┴───────┘
                                        BIT FLAGS
    
    The first 2 bytes of the header are just an arbitrary magic value. The
    header's length field specifies the number of bytes to follow the header in
    the message.
    
    Note that the header's "random" field is used only for UDP discovery, and is
    combined with the "length" field to allow each machine 64 bits of randomness
    in order to guarantee they never add themselves as a peer, with only a very
    low probability of not adding another legitimate peer with the same value.
    
    In the case where a collision does occur, nothing much is lost, as the only
    effect will be that those two machines will not be able to utilize each
    others' caches. Nothing will break. With 64 bits of randomness, ~20k hosts
    would be required for there to be a 1 in 1,000,000,000 chance of at least
    one collision. 
    
    (see https://en.wikipedia.org/wiki/Birthday_problem#Probability_table) */
typedef struct {
    union {
        uint8_t raw[11];
        struct {
            uint16_t magic;     /* Magic value (HEADER_MAGIC). */
            mtype_t  type : 4;  /* Message type. */
            bool     crct : 1;  /* Corrective message flag. */
            bool     unbl : 1;  /* Unable (could not fulfill). */
            bool     rply : 1;  /* Reply (toggle for optional message types). */
            bool     flg3 : 1;  /* Unused. Flag 3. */
            union {
                uint32_t length;    /* Number of bytes following header. */
                uint64_t random;    /* Random identifier for the machine. */
            };
        };
    } header;
    uint8_t data[];
} message_t;

/* File data location for a remote cache. */
typedef struct remote_location {
    UT_hash_handle hh;

    in_addr_t ip;       /* IPv4 address of file owner. Network byte order. */
    char      path[];   /* Filepath. Hash table key. */
} rloc_t;

/* Peer record. */
typedef struct peer_record {
    in_addr_t ip;   /* Peer's IPv4 address. Network byte order. */

    UT_hash_handle hh;
} peer_t;

/* Remote cache state. */
typedef struct {
    rloc_t             *ht;             /* Filepath -> rloc_t hash table. */
    pthread_spinlock_t  ht_lock;        /* Protects HT. */
} rcache_t;

/* User states. Private between users, shared with loader. */
typedef struct {    
    /* Status queues. */
    request_t *head;                /* Only used for teardown. */
    request_t *free;                /* Unused request_t structs. */
    request_t *ready;               /* Ready requests waiting to be executed. */
    request_t *done;                /* Fulfilled requests. */
    request_t *cleanup;             /* Waiting for backend resource cleanup. */

    /* Queue statistics. */
    size_t        queue_depth;  /* Number of entries initially in FREE. */
    atomic_size_t in_flight;    /* Number of entries not in FREE. */

    /* Debug parameters. */
    int        debug_count;     /* In-flight io_uring entries. */
    int        debug_limit;     /* Max DEBUG_COUNT value. 0 = no limit. */
    request_t *bottleneck;      /* Requests that should be processed, but are
                                   bottlenecked by DEBUG_LIMIT. */

    /* Asynchronous IO. */
    struct io_uring ring;

    /* Synchronization. */
    pthread_spinlock_t free_lock;
    pthread_spinlock_t ready_lock;
    pthread_spinlock_t done_lock;
    pthread_spinlock_t cleanup_lock;
} ustate_t;

/* Complete/total cache state. */
typedef struct {
    lcache_t  lcache;   /* Local cache. */
    rcache_t  rcache;   /* Remote cache. */
    uint64_t  random;   /* 64-bit random value from /dev/urandom. */
    uint32_t  n_users;  /* Number of users. */
    uint32_t  qdepth;   /* Queue depth. */
    peer_t   *peers;    /* Iterable hash table of peers. */

    /* Threading info. */
    pthread_t manager_thread;
    pthread_t monitor_thread;
    pthread_t registrar_thread;

    /* Synchronization. */
    pthread_spinlock_t peer_lock;

    /* User-shared memory. */
    ustate_t *ustates;  /* N_USERS + 1 user states. +1 for remote requests. */
} cache_t;

/* Creation/destruction methods. */
cache_t *cache_new(void);
void cache_destroy(cache_t *c);
int cache_init(cache_t *c, size_t capacity, int debug_limit, int queue_depth, int max_unsynced, int n_users);

/* Thread management methods. */
int manager_spawn(cache_t *c);
int monitor_spawn(cache_t *c);
int registrar_spawn(cache_t *c);
void cache_become_manager(cache_t *c);
void cache_become_monitor(cache_t *c);
void cache_become_registrar(cache_t *c);
int cache_start(cache_t *c);

/* Interface methods. */
int cache_get_submit(ustate_t *user, char *path);
int cache_get_reap(ustate_t *user, request_t **out);
int cache_get_reap_wait(ustate_t *user, request_t **out);
void cache_release(ustate_t *user, request_t *request);

#endif
