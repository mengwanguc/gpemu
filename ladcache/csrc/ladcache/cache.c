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

#include "cache.h"
#include "../utils/uthash.h"
#include "../utils/alloc.h"
#include "../utils/fifo.h"
#include "../utils/log.h"
#include <stdbool.h>
#include <pthread.h>
#include <errno.h>
#include <assert.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>

#define PORT_DEFAULT (8080)       /* Default port for TCP and UDP connections. */
#define MAX_QUEUE_REQUESTS (4096) /* Maximum number of queued network requests. */
#define SOCKET_TIMEOUT_S (5)      /* Registrar loop socket timeout. */
#define REGISTER_PERIOD_S (5)     /* Registrar loop broadcast period. */
#define OFF (0)

#define LOG(level, fmt, ...) DEBUG_LOG(SCOPE_INT, level, fmt, ## __VA_ARGS__)
#define MIN(a, b) ((a) > (b) ? (a) : (b))

/* Wrapper for pthread_create. Arguments should be of the same type. COUNTER should
   be a pointer to an atomic_size_t counter for the number of active threads. */
#define PTHREAD_CREATE_DETACH(pid, attr, func, aux)                                                \
    do {                                                                                           \
        int status = pthread_create(pid, attr, func, aux);                                         \
        if (status != 0) {                                                                         \
            LOG(LOG_CRITICAL, "pthread_create failed; %s\n", strerror(status));                    \
            exit(status);                                                                          \
        }                                                                                          \
        if ((status = pthread_detach(*pid)) != 0) {                                                \
            LOG(LOG_CRITICAL, "unable to detach thread %lu; %s\n", *pid, strerror(status));        \
            exit(status);                                                                          \
        }                                                                                          \
    } while (0)

/* Fail if a spin lock does not init*/
#define SPIN_MUST_INIT(spinlock)                                               \
    do {                                                                       \
        int status = pthread_spin_init(spinlock, PTHREAD_PROCESS_SHARED);      \
        assert(!status);                                                       \
    } while (0)

/* Signal a non-zero PID. */
#define KILL_NOT_ZERO(pid, sig)                                                \
    do {                                                                       \
        if (pid != 0) {                                                        \
            kill(pid, sig);                                                    \
        }                                                                      \
    } while (0)


/* --------- */
/*   MISC.   */
/* --------- */

/* Copy IN to OUT, but reformatted to fit shm naming requirements. */
static void
shmify(char *in, char *out, size_t in_length, size_t out_length)
{
    assert(out_length > 0);

    out[0] = '/';
    for (size_t i = 0; i < MIN(in_length, out_length - 1); i++) {
        /* Replace all occurences of '/' with '_'. */
        out[i + 1] = in[i] == '/' ? '_' : in[i];
        if (in[i] == '\0') {
            break;
        }
    }
}

/* An implementation of strncpy_s, i.e., strncpy that doesn't clobber the
   remainder of the destination buffer with zeros. Modified from example given
   at: https://linux.die.net/man/3/strncpy. */
static char *
strncpy_s(char *dest, const char *src, size_t n)
{
    size_t i;

    for (i = 0; i < n && src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
    if (i < n) {
        dest[i] = '\0';
    }

    return dest;
}

/* On success, returns the size of a file in bytes. On failure, returns -errno
   value. */
static off_t
file_get_size(int fd)
{
    struct stat st;
    if (fstat(fd, &st) < 0) {
        return -errno;
    }

    /* Check device type. */
    if (S_ISBLK(st.st_mode)) {
        /* Block device. */
        uint64_t bytes;
        if (ioctl(fd, BLKGETSIZE64, &bytes) != 0) {
            return -errno;
        }
        
        return bytes;
    } else if (S_ISREG(st.st_mode)) {
        return st.st_size;
    }
    
    /* Unknown device type. */
    return -ENODEV;
}


/* --------------------------- */
/*   NETWORK (manager scope)   */
/* --------------------------- */

/* Open a socket to IP (network byte order) on the default port. Returns FD on
   success, -errno on failure. */
static int
network_connect(in_addr_t ip)
{
    struct sockaddr_in peer_addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = ip,
        .sin_port = htons(PORT_DEFAULT)
    };

    /* Open the socket. */
    int peer_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (peer_fd < 0) {
        /* ISSUE: leaking this request. */
        LOG(LOG_ERROR, "Failed to open socket; %s\n", strerror(errno));
        return -errno;
    }
    if (connect(peer_fd, (struct sockaddr *) &peer_addr, sizeof(peer_addr)) < 0) {
        /* ISSUE: leaking this request. */
        LOG(LOG_ERROR, "Failed to connect to %s; %s\n", inet_ntoa(peer_addr.sin_addr), strerror(errno));
        close(peer_fd);
        return -errno;
    }

    return peer_fd;
}

/* Allocates a message_t struct, points OUT to it, and reads a message from FD
   (socket) into it. OUT must only be freed by the user if the function returns
   successfully. Returns 0 on success, -errno on failure. */
static int
network_get_message(int fd, message_t **out)
{
    ssize_t bytes, temp;
    uint32_t len;

    /* Get the request header. */
    message_t *message = malloc(sizeof(message_t));
    if ((bytes = read(fd, (void *) message, sizeof(message_t))) != sizeof(message_t)) {
        if (bytes < 0) {
            LOG(LOG_ERROR, "read failed; %s\n", strerror(errno));
        } else {
            LOG(LOG_WARNING, "Received a message of invalid length (got %ld bytes, expected %ld bytes).\n", bytes, sizeof(message_t));
        }

        free(message);
        return -EBADMSG;
    }

    /* Sanity check. */
    if (message->header.magic != HEADER_MAGIC) {
        LOG(LOG_WARNING, "Received message with invalid header magic (0x%hx, should be 0x%hx).\n", message->header.magic, HEADER_MAGIC);
        free(message);
        return -EBADMSG;
    }
    if ((len = message->header.length) == 0) {  /* Zero is zero regardless of byte order. */
        return 0;
    }

    /* Allocate space for the rest of the message. */
    if ((message = realloc(message, sizeof(message_t) + len)) == NULL) {
        LOG(LOG_ERROR, "Unable to allocate an additional %u bytes for full message.\n", len);
        free(message);
        return -ENOMEM;
    }

    /* Read the rest of the message. */
    bytes = 0;
    while ((temp = read(fd, ((void *) message->data) + bytes, len - bytes)) != EOF && len - bytes > 0) {
        bytes += temp;
    }
    if (len - bytes != 0) {
        LOG(LOG_WARNING, "Expected %u bytes but got %ld.\n", len, bytes);
        free(message);
        return -EBADMSG;
    }

    *out = message;
    return 0;
}

/* Constructs and sends a message to the socket on FD with SIZE bytes of DATA
   as the payload. Does NOT close FD once finished. Returns 0 on sucess and
   -errno on failure. */
static int
network_send_message(mtype_t type, int flags, const void *data, uint32_t size, int fd)
{
    /* Configure and send the header. */
    message_t header;
    memset(&header, 0, sizeof(header));

    /* Configure the header. */
    header.header.type = type;
    header.header.magic = HEADER_MAGIC;
    header.header.length = size;

    /* Send the header. */
    ssize_t bytes;
    if ((bytes = send(fd, (void *) &header, sizeof(header), 0)) != sizeof(header)) {
        if (bytes < 0) {
            LOG(LOG_ERROR, "Failed to send header; %s\n", strerror(errno));
            return -errno;
        } else {
            LOG(LOG_ERROR, "Failed to send entire header (%ld/%lu bytes sent).\n", bytes, sizeof(header));
            return -EAGAIN;
        }
    }
    
    /* Send the data. Stop sending once all sent, or on failure to send. */
    bytes = 0;
    ssize_t temp;
    while ((temp = send(fd, data + bytes, size, 0)) > 0 && (bytes += temp) != size) {}
    if (bytes != size) {
        if (temp < 0) {
            LOG(LOG_ERROR, "Failed to send payload; %s\n", strerror(errno));
            return -errno;
        } else {
            LOG(LOG_ERROR, "Failed to send entire payload (%ld/%u bytes sent) MAX_QUEUE_REQUESTS = %d.\n",
                bytes, size, MAX_QUEUE_REQUESTS);
            return -EAGAIN;
        }
    }

    return 0;
}


/* --------------------------- */
/*   MONITOR (manager scope)   */
/* --------------------------- */

/* Announce our existence to other members of the distributed cache. Returns 0
   on success and -errno on failure. */
static int
cache_register(cache_t *c)
{
    /* Create broadcast socket. */
    int broadcast_fd = socket(AF_INET, SOCK_DGRAM, 0);
    int broadcast = 1;
    int status = setsockopt(broadcast_fd,
                            SOL_SOCKET,
                            SO_BROADCAST,
                            &broadcast,
                            sizeof(broadcast));
    if (status < 0) {
        LOG(LOG_ERROR, "Failed to configure socket for broadcast; %s\n", strerror(errno));
        return -errno;
    }

    /* Registration "hello" message with reply required. */
    message_t header = {
        .header.magic = HEADER_MAGIC,
        .header.type = TYPE_HLLO,
        .header.crct = false,
        .header.unbl = false,
        .header.rply = true,
        .header.flg3 = false,
        .header.random = c->random
    };

    /* Broadcast the message. */
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_BROADCAST),
        .sin_port = htons(PORT_DEFAULT)
    };
    ssize_t bytes = sendto(broadcast_fd,
                           (void *) &header,
                           sizeof(message_t),
                           FLAG_NONE,
                           (struct sockaddr *) &addr,
                           sizeof(addr));
    if (bytes != sizeof(message_t)) {
        if (bytes < 0) {
            LOG(LOG_ERROR, "Failed to broadcast hello; %s\n", strerror(errno));
            return -errno;
        } else {
            LOG(LOG_ERROR, "Failed to broadcast entire message (%ld/%lu bytes).\n", bytes, sizeof(message_t));
            return -EBADMSG;
        }
    }

    close(broadcast_fd);
    return 0;
}

/* Send a message to every known peer with our list of newly-cached files.
   Returns 0 on success, -errno on failure. Assumes that there is at least one
   unsynced filepath in the list. */
static int
cache_sync(cache_t *c)
{
    /* No-op if the unsynced list is already empty. */
    lloc_t *loc = c->lcache.unsynced;
    if (loc == NULL) {
        return 0;
    }

    /* Determine the sum of all filepath lengths. */
    uint32_t n_entries = 0;
    size_t payload_len = sizeof(uint32_t);
    do {
        payload_len += strlen(loc->path) + 1; /* +1 for '\0' byte. */
        n_entries++;
    } while ((loc = loc->next) != NULL);

    /* Allocate our message payload. */
    char *payload = malloc(payload_len);
    if (payload == NULL) {
        LOG(LOG_ERROR, "Unable to allocate %lu bytes for sync payload.\n", payload_len);
        return -ENOMEM;
    }
    *((uint32_t *) payload) = n_entries;

    /* Write all of the filepaths and clear the unsynced list. */
    char *fp_dest = payload + sizeof(uint32_t);
    do {
        QUEUE_POP(c->lcache.unsynced, next, prev, loc);
        assert(fp_dest + strlen(loc->path) + 1 <= payload + payload_len);
        strncpy_s(fp_dest, loc->path, MAX_PATH_LEN + 1);
        fp_dest += strlen(loc->path) + 1; /* +1 for '\0' byte. */
    } while (c->lcache.unsynced != NULL);

    /* Send the message to everyone we know. */
    pthread_spin_lock(&c->peer_lock);
    peer_t *peer, *tmp;
    HASH_ITER(hh, c->peers, peer, tmp) {
        /* Open the socket. */
        int peer_fd = network_connect(peer->ip);
        if (peer_fd < 0) {
            LOG(LOG_ERROR, "network_connect failed; %s\n", strerror(-peer_fd));
            pthread_spin_unlock(&c->peer_lock);
            free(payload);
            return peer_fd;
        }

        /* Send the message. */
        LOG(LOG_DEBUG, "Sending SYNC to %s with %u files.\n", inet_ntoa((struct in_addr) {.s_addr = peer->ip}), n_entries);
        int status = network_send_message(TYPE_SYNC,
                                          FLAG_NONE,
                                          (void *) payload,
                                          payload_len,
                                          peer_fd);
        if (status < 0) {
            LOG(LOG_ERROR, "Failed to send sync message; %s\n", strerror(-status));
            pthread_spin_unlock(&c->peer_lock);
            free(payload);
            return status;
        }
        close(peer_fd);
    };
    pthread_spin_unlock(&c->peer_lock);

    c->lcache.n_unsynced = 0;
    free(payload);
    return 0;
}

/* Handle a file request by sending the requested file data. Returns 0 on
   success, -errno on failure. */
static int
monitor_handle_request(message_t *message, cache_t *c, int fd)
{
    struct sockaddr_in addr;
    socklen_t addr_size = sizeof(addr);
    if (getpeername(fd, (struct sockaddr *) &addr, &addr_size) < 0) {
        LOG(LOG_ERROR, "getpeername failed; %s\n", strerror(-errno));
        return -errno;
    }

    /* TODO. Verify the filepath is the correct length (i.e., \0 terminated). */
    char *path = (char *) message->data;
    LOG(LOG_DEBUG, "Received request from %s for \"%s\".\n", inet_ntoa(addr.sin_addr), message->data);

    /* Try to retrieve the requested file from our cache. If uncached, reply
       that we are unable to fulfill their request. */
    lloc_t *loc;
    HASH_FIND_STR(c->lcache.ht, (char *) message->data, loc);
    if (loc == NULL) {
        LOG(LOG_WARNING, "File \"%s\" is not cached.\n", message->data);
        return network_send_message(TYPE_RSPN, FLAG_UNBL, NULL, 0, fd);
    }

    /* Otherwise, reply with our cached file data. */
    LOG(LOG_DEBUG, "Sending %s %s (%u bytes).\n", inet_ntoa(addr.sin_addr), path, (uint32_t) loc->size);
    return network_send_message(TYPE_RSPN, FLAG_NONE, loc->data, loc->size, fd);
}

/* Handle a directory sync message. */
static int
monitor_handle_sync(message_t *message, cache_t *c, int fd)
{
    /* TODO. Verify filepaths are valid ('\0' terminated, etc.) */

    /* Figure out who we're talking to. */
    struct sockaddr_in addr;
    socklen_t addr_size = sizeof(addr);
    if (getpeername(fd, (struct sockaddr *) &addr, &addr_size) < 0) {
        return -errno;
    }
    in_addr_t ip = addr.sin_addr.s_addr; /* Host order. */

    /* Add their filepaths to the remote cache directory. */
    uint32_t n_entries = *((uint32_t *) message->data);
    char *filepath = ((char *) message->data) + sizeof(uint32_t);
    for (uint32_t i = 0; i < n_entries; i++) {
        /* Check string is in valid memory range. */
        size_t fp_len = strlen(filepath);
        if ((void *) (filepath + fp_len + 1) >
            (void *) (message->data + message->header.length)) {
            LOG(LOG_WARNING, "Invalid string length.\n");
            return -ERANGE;
        }

        /* Prepare the hash table entry.
        
           NOTE: it would be more efficient to malloc a single chunk of memory
           for all entries, however this would be more difficult to track. */
        rloc_t *loc = malloc(sizeof(rloc_t) + fp_len + 1);
        if (loc == NULL) {
            LOG(LOG_ERROR, "Failed to allocate lloc_t struct.\n");
            return -ENOMEM;
        }
        loc->ip = ip;
        strncpy_s(loc->path, filepath, fp_len + 1);

        /* Add to the remote cache directory. */
        LOG(LOG_DEBUG, "Adding \"%s\" to the remote cache directory\n", loc->path);
        HASH_ADD_STR(c->rcache.ht, path, loc);

        /* Move to the next filepath. */
        filepath += strlen(filepath) + 1;
    }
    LOG(LOG_INFO, "Received SYNC from %s with %u files.\n", inet_ntoa(addr.sin_addr), n_entries);

    return 0;
}

/* Arguments to monitor_handle_connection. */
struct monitor_handle_connection_args {
    cache_t *c;
    int peer_fd;
};

/* Handles a remote read request. */
static void *
monitor_handle_connection(void *args)
{
    /* Get the arguments passed to us. */
    cache_t *c = ((struct monitor_handle_connection_args *) args)->c;
    int peer_fd = ((struct monitor_handle_connection_args *) args)->peer_fd;

    /* Read the initial message from the socket. */
    message_t *message;
    int status = network_get_message(peer_fd, &message);
    if (status < 0) {
        LOG(LOG_WARNING, "network_get_message failed; %s\n", strerror(-status));
        close(peer_fd);
        return NULL;
    }

    /* Dispatch for the proper handler for this message type. */
    switch (message->header.type) {
        case TYPE_RQST: monitor_handle_request(message, c, peer_fd); break;
        case TYPE_SYNC: monitor_handle_sync(message, c, peer_fd); break;
        default:
            LOG(LOG_WARNING, "Received an invalid message; type = 0x%hx.\n", message->header.type);
    }

    free(message);
    close(peer_fd);
    return NULL;
}

/* Monitor main loop. Handles all incoming remote read requests. Should never
   return when running correctly. On failure returns negative errno value. */
static void *
monitor_loop(void *args)
{
    cache_t *c = (cache_t *) args;

    /* Open the listening socket. */
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        return NULL;
    }

    /* Allow address to be re-used (needed?) */
    int opt = 1;
    if (setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        LOG(LOG_CRITICAL, "Failed to configure socket; %s\n", strerror(errno));
        return NULL;
    }

    /* Bind to PORT_DEFAULT. */
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_ANY),
        .sin_port = htons(PORT_DEFAULT)
    };
    socklen_t addr_len = sizeof(addr);
    if (bind(lfd, (struct sockaddr *)&addr, addr_len) < 0) {
        LOG(LOG_CRITICAL, "Failed to bind socket; %s\n", strerror(errno));
        return NULL;
    }

    /* Start listening. */
    if (listen(lfd, MAX_QUEUE_REQUESTS)) {
        LOG(LOG_CRITICAL, "Failed to listen to socket; %s\n", strerror(errno));
        return NULL;
    }

    /* Handle all incoming connections. */
    while (true) {
        int cfd = accept(lfd, (struct sockaddr *) &addr, (socklen_t *) &addr_len);
        if (cfd >= 0) {
            /* Prepare arguments. */
            struct monitor_handle_connection_args *conn_args = malloc(
                sizeof(struct monitor_handle_connection_args)
            );
            if (conn_args == NULL) {
                LOG(LOG_CRITICAL, "failed to allocate arguments for connection handler.\n");
                return NULL;
            }
            conn_args->c = c;
            conn_args->peer_fd = cfd;

            /* This thread will terminate gracefully on its own and we don't
               need to track it. */
            pthread_t _;
            PTHREAD_CREATE_DETACH(&_, NULL, monitor_handle_connection, conn_args);
        }
    }

    NOT_REACHED();
    return 0;
}

/* Spawns a new thread running the monitor loop. Returns 0 on success, -errno on
   failure. */
int
monitor_spawn(cache_t *c)
{
    return -pthread_create(&c->monitor_thread, NULL, monitor_loop, c);
}


/* ----------------------------- */
/*   REGISTRAR (manager scope)   */
/* ----------------------------- */

/* Loop to listen for incoming UDP registration datagrams. */
static void *
registrar_loop(void *args)
{
    cache_t *c = (cache_t *) args;
    int status;

    /* Create listening socket. */
    int sfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sfd < 0) {
        LOG(LOG_CRITICAL, "Failed to create registrar listening socket; %s\n", strerror(errno));
        return NULL;
    }

    /* Set socket timeout to SOCKET_TIMEOUT_MS milliseconds. */
    struct timeval tv = {
        .tv_sec = SOCKET_TIMEOUT_S,
        .tv_usec = 0,
    };
    if ((status = setsockopt(sfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv))) < 0) {
        LOG(LOG_CRITICAL, "Failed to set registrar socket timeout; %s\n", strerror(errno));
        goto fail;
    }

    /* Accept all incoming connections. */
    socklen_t addr_len = sizeof(struct sockaddr_in);
    struct sockaddr_in server_addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_ANY),
        .sin_port = htons(PORT_DEFAULT)
    };
    if ((status = bind(sfd, (const struct sockaddr *) &server_addr, addr_len)) < 0) {
        LOG(LOG_CRITICAL, "Failed to bind socket; %s\n", strerror(errno));
        goto fail;
    }

    /* Continually await new datagrams, and respond in kind upon reception. */
    message_t header;
    struct sockaddr_in client_addr;
    time_t last_bc = 0, now;
    while (true) {
        /* Broadcast a registration message every REGISTRATION_PERIOD_MS
           milliseconds to ensure no peers are missed by drops, etc. */
        if ((now = time(NULL)) - last_bc >= REGISTER_PERIOD_S) {
            cache_register(c);
            last_bc = now;
        }

        /* Receive new datagram. */
        ssize_t bytes = recvfrom(sfd,
                                 (void *) &header,
                                 sizeof(message_t),
                                 FLAG_NONE,
                                 (struct sockaddr *) &client_addr,
                                 &addr_len);
        if (bytes != sizeof(message_t)) {
            if (bytes < 0) {
                /* Ignore timeout failures. They're necessary to ensure we can
                   continue to broadcast our existence periodically. */
                if (errno != EAGAIN) {
                    LOG(LOG_WARNING, "Failed to receive datagram; %s\n", strerror(errno));
                }
            } else {
                LOG(LOG_WARNING, "Received incomplete header (%ld/%lu bytes).\n", bytes, sizeof(message_t));
            }
            continue;
        }

        /* Check validity of message. */
        if (header.header.magic != HEADER_MAGIC ||
            header.header.type != TYPE_HLLO ||
            header.header.random == c->random) {
            continue;
        }

        /* Add sender to our directory if it isn't already in it. */
        peer_t *peer = NULL;
        uint32_t peer_ip = client_addr.sin_addr.s_addr;
        HASH_FIND_INT(c->peers, &peer_ip, peer);
        if (peer == NULL) {
            peer_t *peer = malloc(sizeof(peer_t));
            if (peer == NULL) {
                LOG(LOG_CRITICAL, "Failed to allocate peer record.\n");
                goto fail;
            }
            peer->ip = peer_ip;
            HASH_ADD_INT(c->peers, ip, peer);
            LOG(LOG_INFO, "Added %s to set of peers\n", inet_ntoa(client_addr.sin_addr));
        }

        /* If reply is required, send a no-reply-required response by re-using
           the client address and their message. */
        if (header.header.rply) {
            header.header.rply = false;
            header.header.random = c->random;
            if ((bytes = sendto(sfd,
                                (void *) &header,
                                sizeof(message_t),
                                FLAG_NONE,
                                (struct sockaddr *) &client_addr,
                                sizeof(client_addr)) != sizeof(message_t))) {
                if (bytes < 0) {
                    LOG(LOG_WARNING, "Failed to send registration reply; %s\n", strerror(errno));
                } else {
                    LOG(LOG_WARNING, "Failed to send entire registration reply (%ld/%lu bytes).\n", bytes, sizeof(message_t));
                }
            }
        }
    }

   fail:
    close(sfd);
    return NULL;
}

/* Spawn a thread running the registrar loop. */
int
registrar_spawn(cache_t *c)
{
    return -pthread_create(&c->registrar_thread, NULL, registrar_loop, c);
}


/* ----------------------------------------- */
/*   LOCAL CACHE INTERFACE (manager scope)   */
/* ----------------------------------------- */

/* Checks the local cache for PATH. Returns TRUE if contained, FALSE if not. */
static bool
cache_local_contains(lcache_t *lc, char *path)
{
    lloc_t *loc = NULL;
    HASH_FIND_STR(lc->ht, path, loc);

    return (loc != NULL);
}

/* Caches SIZE bytes of DATA using PATH as the key. Copies DATA into a newly
   allocated shm object. Returns 0 on success, and a negative errno value on
   failure. */
static int
cache_local_store(lcache_t *lc, char *path, uint8_t *data, size_t size)
{
    /* Verify we can fit this in the cache. */
    if (lc->used + size > lc->capacity) {
        LOG(LOG_DEBUG, "%s (%lu byte) is too big to fit in local cache.\n", path, size);
        return -E2BIG;
    }

    /* Get a new location record. */
    lloc_t *loc = malloc(sizeof(lloc_t));
    if (loc == NULL) {
        LOG(LOG_ERROR, "Failed to allocate lloc_t struct.\n");
        return -ENOMEM;
    }

    /* Allocate shared memory. */
    loc->shm_size = size;
    shmify(path, loc->shm_path, MAX_PATH_LEN + 1, MAX_SHM_PATH_LEN + 1);
    loc->shm_fd = shm_alloc(loc->shm_path, &loc->data, loc->shm_size);
    if (loc->shm_fd < 0) {
        int status = loc->shm_fd;
        free(loc);
        return status;
    }

    /* Copy data in. */
    memcpy(loc->data, data, size);

    /* Insert into hash table. */
    strncpy_s(loc->path, path, MAX_PATH_LEN + 1);
    HASH_ADD_STR(lc->ht, path, loc);

    return 0;
}

/* Fill a request with data from the local cache. Returns 0 on success, -errno
   on failure. */
static int
cache_local_load(lcache_t *lc, request_t *request)
{
    /* Get the location record of the cached file. */
    lloc_t *loc = NULL;
    HASH_FIND_STR(lc->ht, request->path, loc);
    if (loc == NULL) {
        LOG(LOG_ERROR, "Attempted to load uncached file: \"%s\"\n", request->path);
        return -ENODATA;
    }

    /* Fill the request. */
    request->size = loc->size;
    request->shm_size = loc->shm_size;
    request->_ldata = loc->data;
    request->_lfd_shm = loc->shm_fd;
    request->_skip_clean = true; /* Don't purge this entry once we're done with
                                    this particularly request. */

    return 0;
}


/* ------------------------------------------ */
/*   REMOTE CACHE INTERFACE (manager scope)   */
/* ------------------------------------------ */

/* Checks the remote cache for PATH. Returns TRUE if contained, FALSE if not. */
static bool
cache_remote_contains(rcache_t *rc, char *path)
{
    rloc_t *loc = NULL;
    HASH_FIND_STR(rc->ht, path, loc);
    
    return (loc != NULL);
}

/* Arguments to cache_remote_load. */
struct cache_remote_load_args {
    cache_t   *c;
    rcache_t  *rc;
    ustate_t  *user;
    request_t *request;
};

/* Thread target to request a file from a peer. Should be passed a malloc'd
   cache_remote_load_args struct, which will be freed once arguments have been
   parsed. Assumes that the caller has already removed REQUEST from USER's ready
   queue. The completed request will be placed into USER's done queue. */
static void *
cache_remote_load(void *args)
{
    /* Get the arguments passed to us. */
    struct cache_remote_load_args *load_args = (struct cache_remote_load_args *) args;
    rcache_t *rc = load_args->rc;
    ustate_t *user = load_args->user;
    request_t *request = load_args->request;
    free(args);

    /* Find who to request the file from. */
    rloc_t *loc = NULL;
    HASH_FIND_STR(rc->ht, request->path, loc);
    if (loc == NULL) {
        /* ISSUE: leaking this request. */
        LOG(LOG_ERROR, "Tried to load file that's not in the remote cache; %s\n", request->path);
        request->status = -ENOENT;
        goto done;
    }

    /* Connect to them. */
    int peer_fd = network_connect(loc->ip);
    if (peer_fd < 0) {
        LOG(LOG_ERROR, "network_connect failed; %s\n", strerror(-peer_fd));
        request->status = peer_fd;
        goto done;
    }

    /* Send them a request for the file. */
    LOG(LOG_DEBUG, "Requesting file \"%s\" from %s.\n", request->path, inet_ntoa((struct in_addr) {.s_addr = loc->ip}));
    int status = network_send_message(TYPE_RQST,
                                      FLAG_NONE,
                                      request->path,
                                      strlen(request->path) + 1,
                                      peer_fd);
    if (status < 0) {
        LOG(LOG_ERROR, "Failed to send message; %s\n", strerror(-status));
        request->status = status;
        close(peer_fd);
        goto done;
    }

    /* Wait for a response. */
    message_t *response;
    status = network_get_message(peer_fd, &response);
    close(peer_fd); /* Socket use finished here. */
    if (status < 0) {
        LOG(LOG_ERROR, "network_get_message failed; %s\n", strerror(-status));
        request->status = status;
        goto done;
    }

    /* Make sure we got a sensible response. */
    if (response->header.type != TYPE_RSPN) {
        LOG(LOG_WARNING, "Received an incorrect message type (type = 0x%hx)\n", response->header.type);
        request->status = status;
        free(response);
        goto done;
    }

    /* Was the peer unable to fulfill the request? */
    if (response->header.unbl) {
        LOG(LOG_WARNING, "%s unable to fulfill request for \"%s\".\n", inet_ntoa((struct in_addr) {.s_addr = loc->ip}), request->path);
        request->status = status;
        free(response);
        goto done;
    }

    /* Allocate an shm object for the file data. Note it would be more efficient
       to have read() read directly into the shm object, however this would
       complicate the get_message interface and so we'll just eat the copy cost
       for now. */
    request->size = response->header.length;
    request->shm_size = request->size;
    shmify(request->path, request->shm_path, MAX_PATH_LEN + 1, MAX_SHM_PATH_LEN + 1);
    request->_lfd_shm = shm_alloc(request->shm_path, &request->_ldata, request->shm_size);
    memcpy(request->_ldata, response->data, request->shm_size);
    
    LOG(LOG_DEBUG, "Received \"%s\" (%u bytes) from %s.\n", request->path, response->header.length, inet_ntoa((struct in_addr) {.s_addr = loc->ip}));

   done:
    assert(request->path[0] != '\0');
    QUEUE_PUSH_SAFE(user->done, &user->done_lock, next, prev, request);
    free(response);
    return NULL;
}


/* --------------------------- */
/*   MANAGER (manager scope)   */
/* --------------------------- */

/* Submit an IO request to io_uring. Returns 0 on success, -errno on failure. */
static int
manager_submit_io(ustate_t *ustate, request_t *r)
{
    LOG(LOG_DEBUG, "Loading \"%s\" from storage.\n", r->path);

    /* Open the file. */
    r->_lfd_file = open(r->path, O_RDONLY | __O_DIRECT);
    if (r->_lfd_file < 0) {
        LOG(LOG_ERROR, "Failed to open \"%s\"; %s\n", r->path, strerror(errno));
        return -errno;
    }

    /* Get the size of the file, rounding the size up to the nearest multiple of
       4KB for O_DIRECT compatibility. */
    off_t size = file_get_size(r->_lfd_file);
    if (size < 0) {
        LOG(LOG_ERROR, "file_get_size failed.\n");
        return (int) size;
    }
    r->shm_size = (((size_t) size) | 0xFFF) + 1;
    r->size = (size_t) size;

    /* Create buffer using shm. */
    r->_lfd_shm = shm_alloc(r->shm_path, &r->_ldata, r->shm_size);
    if (r->_lfd_shm < 0) {
        LOG(LOG_ERROR, "shm_alloc failed; %s\n", strerror(-r->_lfd_shm));
        close(r->_lfd_file);
        return r->_lfd_shm;
    }

    /* Tell io_uring to read the file into the buffer. */
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ustate->ring);
    io_uring_prep_read(sqe, r->_lfd_file, r->_ldata, r->shm_size, 0);
    io_uring_sqe_set_data(sqe, r);
    io_uring_submit(&ustate->ring);

    return 0;
}

/* Check whether USTATE has any backend resources to be cleaned up. If resources
   exist, clean up a single request_t struct per call. Returns TRUE when a
   resource was cleaned up, FALSE when there was nothing to clean up. */
static bool
manager_check_cleanup(cache_t *c, ustate_t *ustate)
{
    /* Check if there's anything in the queue. */
    request_t *to_clean = NULL;
    QUEUE_POP_SAFE(ustate->cleanup, &ustate->cleanup_lock, next, prev, to_clean);
    if (to_clean == NULL) {
        return false;
    }

    assert(to_clean->path[0] != '\0');

    /* Check if it should be cleaned up (not exempt, not in an error state). */
    if (!to_clean->_skip_clean && !to_clean->status) {
        LOG(LOG_DEBUG, "Deep cleaning \"%s\" entry (%s).\n", to_clean->path, to_clean->shm_path);
        munmap(to_clean->_ldata, to_clean->shm_size);
        close(to_clean->_lfd_shm);
        shm_unlink(to_clean->shm_path);

        /* If we loaded from the remote cache we'll have never opened a local
           file, and so _lfd_file will still be 0. */
        if (to_clean->_lfd_file != 0) {
            close(to_clean->_lfd_file);
        }
    }

    /* Wipe it. */
    memset(to_clean, 0, sizeof(request_t));

    /* Move it to the free queue. */
    QUEUE_PUSH_SAFE(ustate->free, &ustate->free_lock, next, prev, to_clean);
    atomic_fetch_sub(&ustate->in_flight, 1);

    return true;
}

/* Check whether USTATE has a pending request and execute it if it does. Returns
   0 on sucess, -errno on failure. */
static int
manager_check_ready(cache_t *c, ustate_t *ustate)
{
    request_t *pending = NULL;

    /* Bottleneck is a debug setting. Process bottlenecked requests must be
       processed first to maintain FIFO order. */
    if (ustate->debug_limit != OFF && ustate->debug_count < ustate->debug_limit) {
        QUEUE_POP(ustate->bottleneck, next, prev, pending);
    }

    /* Pop from the regular queue if not processing a bottlenecked request. */
    if (pending == NULL) {
        QUEUE_POP_SAFE(ustate->ready, &ustate->ready_lock, next, prev, pending);
        if (pending == NULL) {
            return 0;
        }
    }

    /* Check the local cache. */
    if (cache_local_contains(&c->lcache, pending->path)) {
        pending->status = cache_local_load(&c->lcache, pending);
        if (pending->status < 0) {
            LOG(LOG_ERROR, "cache_local_load failed; %s\n", strerror(-pending->status));
        }
        assert(pending->path[0] != '\0');
        assert(pending->shm_size != 0);

        QUEUE_PUSH_SAFE(ustate->done, &ustate->done_lock, next, prev, pending);

        return 0;
    }

    /* Check the remote cache. */
    if (cache_remote_contains(&c->rcache, pending->path)) {
        /* Prepare arguments for cache_remote_load. */
        struct cache_remote_load_args *args = malloc(sizeof(struct cache_remote_load_args));
        if (args == NULL) {
            return -ENOMEM;
        }
        args->c = c;
        args->rc = &c->rcache;
        args->request = pending;
        args->user = ustate;

        LOG(LOG_DEBUG, "Spawning cache_remote_load thread for \"%s\".\n", pending->path);

        /* Spawn a thread to handling requesting the file from the peer. It will
           take care of itself and doesn't require management. */
        pthread_t _;
        PTHREAD_CREATE_DETACH(&_, NULL, cache_remote_load, args);

        return 0;
    }

    /* File neither cached remotely nor locally. Unless DEBUG_LIMIT enabled,
       submit this request to be loaded locally via io_uring. */
    if (ustate->debug_limit != OFF && ustate->debug_count >= ustate->debug_limit) {
        QUEUE_PUSH(ustate->bottleneck, next, prev, pending);
        return 0;
    }

    int status = manager_submit_io(ustate, pending);
    if (status < 0) {
        pending->status = status;
        assert(pending->path[0] != '\0');
        QUEUE_PUSH_SAFE(ustate->done, &ustate->done_lock, next, prev, pending);
        LOG(LOG_ERROR, "manager_submit_io failed; %s\n", strerror(-status));
        return status;
    }
    ustate->debug_count++;

    return 0;
}

/* Check if any storage requests have completed their IO. Note that the network
   monitor handles completed network requests. Returns 0 on sucess, -errno on
   failure. */
static void
manager_check_done(cache_t *c, ustate_t *ustate)
{
    /* Drain the io_uring completion queue into our completion queue. Using
       peek (instead of wait) to ensure the check is non-blocking. */
    struct io_uring_cqe *cqe;
    while (!io_uring_peek_cqe(&ustate->ring, &cqe)) {
        request_t *request = io_uring_cqe_get_data(cqe);
        io_uring_cqe_seen(&ustate->ring, cqe);
        ustate->debug_count--;
        if (cqe->res < 0) {
            /* If io_uring's called to read() failed. */
            LOG(LOG_ERROR, "asynchronous read failed; %s\n", strerror(-cqe->res));
            request->status = cqe->res;
            goto skip_cache;
        }

        /* Try to cache this file. */
        if (c->lcache.used + request->shm_size <= c->lcache.capacity) {
            lloc_t *loc = malloc(sizeof(lloc_t));
            if (loc == NULL) {
                LOG(LOG_ERROR, "Failed to allocate lloc_t struct.\n");
                request->status = -ENOMEM;
                goto skip_cache;
            }

            /* Prepare the location record and append it to the list of unsynced
               filepaths for this epoch. */
            *loc = (lloc_t) {
                .data = request->_ldata,
                .size = request->size,
                .shm_fd = request->_lfd_shm,
                .shm_size = request->shm_size
            };
            strncpy_s(loc->path, request->path, MAX_PATH_LEN + 1);
            strncpy_s(loc->shm_path, request->shm_path, MAX_SHM_PATH_LEN + 1);

            /* Add to the hash table indexed by PATH. */
            HASH_ADD_STR(c->lcache.ht, path, loc);
            c->lcache.used += loc->shm_size;
            request->_skip_clean = true;

            LOG(LOG_DEBUG, "Added \"%s\" to local cache; marked to skip cleanup.\n", loc->path);

            /* Add to list of filenames to be synchronized. */
            QUEUE_PUSH(c->lcache.unsynced, next, prev, loc);
            c->lcache.n_unsynced++;
        }

       skip_cache:
        assert(request->path[0] != '\0');
        QUEUE_PUSH_SAFE(ustate->done, &ustate->done_lock, next, prev, request);
    }
}

/* Manager main loop. Handles all pending requests. */
static void *
manager_loop(void *args)
{
    cache_t *c = (cache_t *) args;
    size_t prev_length;
    size_t idle_iters = 0;
    uint64_t i = 0;

    /* Loop round-robin through the user ustates and check for pending and
       completed requests that require status queue updates. */
    while (true) {
        ustate_t *ustate = &c->ustates[i++ % c->n_users];
        /* Check if we need to sync our newly cached files with peers. We do
           this when we've either reached the submission threshold, or when
           we've looped many times without caching anything new. A threshold of
           zero indicates no limit. */
        if ((c->lcache.n_unsynced >= c->lcache.threshold && c->lcache.threshold > 0) ||
            (idle_iters > (MAX_IDLE_ITERS) && c->lcache.n_unsynced > 0)) {
            LOG(LOG_DEBUG, "Syncing %lu filepaths.\n", c->lcache.n_unsynced);
            idle_iters = 0;
            cache_sync(c);
        }

        prev_length = c->lcache.n_unsynced;
        manager_check_ready(c, ustate);
        manager_check_done(c, ustate);

        /* Clean up all entries in this user state. Cleaning only a single entry
           will cause a significant bottleneck to occur when there the requests
           are saturated. */
        while (manager_check_cleanup(c, ustate))

        /* Reset the idle count if we've got new unsynced data. Otherwise,
           continue to increment it. */
        if (c->lcache.n_unsynced > prev_length) {
            idle_iters = 0;
        } else {
            idle_iters++;
        }
    }

    NOT_REACHED();
    return NULL;
}

/* Spawns a new thread running the manager loop. Returns 0 on success, -errno on
   failure. */
int
manager_spawn(cache_t *c)
{
    return -pthread_create(&c->manager_thread, NULL, manager_loop, c);
}


/* ---------------------------------- */
/*   GENERIC INTERFACE (user scope)   */
/* ---------------------------------- */

/* Become the manager thread. Does not return. */
void
cache_become_manager(cache_t *c)
{
    c->manager_thread = getpid();
    manager_loop((void *) c);
}

/* Become the monitor thread. Does not return. */
void
cache_become_monitor(cache_t *c)
{
    c->monitor_thread = getpid();
    monitor_loop((void *) c);
}

/* Become the registrar thread. Does not return. */
void
cache_become_registrar(cache_t *c)
{
    c->registrar_thread = getpid();
    registrar_loop((void *) c);
}


/* Spawn the manager, the monitor, and the registrar. Returns 0 on success,
   -errno on failure. */
int
cache_start(cache_t *c)
{
    int status;
    if ((status = manager_spawn(c)) < 0) {
        return status;
    }
    if ((status = monitor_spawn(c)) < 0) {
        kill(c->manager_thread, SIGKILL);
        return status;
    }
    if ((status = registrar_spawn(c)) < 0) {
        kill(c->manager_thread, SIGKILL);
        kill(c->monitor_thread, SIGKILL);
        return status;
    }

    return 0;
}

/* Submit a request for the file at PATH to be loaded for USER. Returns 0 on
   success, -errno on failure. */
int
cache_get_submit(ustate_t *user, char *path)
{
    /* Generate request. */
    request_t *request = NULL;
    atomic_fetch_add(&user->in_flight, 1);
    QUEUE_POP_SAFE(user->free, &user->free_lock, next, prev, request);
    if (request == NULL) {
        LOG(LOG_DEBUG, "Free queue is empty; no request_t structs available.\n");
        return -EAGAIN; /* Try again once completed requests have been freed. */
    }
    memset(request, 0, sizeof(request_t));
    strncpy_s(request->path, path, MAX_PATH_LEN + 1);
    shmify(request->path, request->shm_path, MAX_PATH_LEN + 1, MAX_SHM_PATH_LEN + 1);

    /* Submit request to the monitor. */
    QUEUE_PUSH_SAFE(user->ready, &user->ready_lock, next, prev, request);
    
    return 0;
}

/* Reap a completed request for USER. Points OUT to a completed request. Returns
   0 on sucess, -errno on failure. Unless -EAGAIN is retured, OUT will point to
   a request_t struct (successful or not) taken from the done queue. */
int
cache_get_reap(ustate_t *user, request_t **out)
{
    int status = 0;

    /* Try to get a completed request. */
    request_t *r = NULL;
    QUEUE_POP_SAFE(user->done, &user->done_lock, next, prev, r);
    if (r == NULL) {
        return -EAGAIN; /* Try again once request has been fulfilled. */
    }

    /* Check the string is non-empty. */
    assert(r->path[0] != '\0');

    /* Check if the request failed. */
    if (r->status < 0) {
        LOG(LOG_WARNING, "Reaped a failed request for \"%s\"; %s\n", r->path, strerror(-r->status));
        status = r->status;
        goto done;
    }

    /* Open the shm object. */
    if ((r->ufd_shm = shm_open(r->shm_path, O_RDONLY, S_IRUSR)) < 0) {
        LOG(LOG_ERROR, "shm_open failed; \"%s\"; %s\n", r->shm_path, strerror(errno));
        status = -errno;
        goto done;
    }

    /* Create the mmap. */
    if ((r->udata = mmap(NULL, r->shm_size, PROT_READ, MAP_PRIVATE, r->ufd_shm, 0)) == (void *) -1LL) {
        LOG(LOG_ERROR, "mmap failed (ufd_shm=%d, shm_size=%lu, status=%d); %s\n", r->ufd_shm, r->shm_size, r->status, strerror(errno));
        status = -errno;
        goto done;
    }

   done:
    *out = r;
    return status;
}

/* Spin on cache_get_reap until an entry becomes ready. Returns 0 on success,
   -errno on failure. */
int
cache_get_reap_wait(ustate_t *user, request_t **out)
{
    int status;
    while ((status = cache_get_reap(user, out)) == -EAGAIN) sched_yield();
    return status;
}

/* Release REQUEST, free user resources and move the request into the cleanup
   queue (for backend resources to be evaluated and reclaimed). */
void
cache_release(ustate_t *user, request_t *request)
{
    if (request == NULL) {
        return;
    }

    if (request->status == 0) {
        /* ISSUE: Leaking resources on failures. Needs to be more granular. */
        if (munmap(request->udata, request->shm_size)) {
            LOG(LOG_CRITICAL, "munmap failed; %s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }
        if (close(request->ufd_shm)) {
            LOG(LOG_CRITICAL, "close failed; %s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }
    }
    QUEUE_PUSH_SAFE(user->cleanup, &user->cleanup_lock, next, prev, request);
}


/* --------------------------- */
/*   ALLOCATION (user scope)   */
/* --------------------------- */

/* Get a cache_t struct using shared memory. Returns NULL on failure. */
cache_t *
cache_new(void)
{
    return mmap_alloc(sizeof(cache_t));
}

/* Destroy a complete cache. Allows for the destruction of both partially and
   fully allocated caches. */
void
cache_destroy(cache_t *c)
{
    /* TODO. Update to clean up rloc_t and lloc_t structs from hash tables, and
             free all of the peer_t records. */

    if (c == NULL) {
        return;
    }

    /* Free all of the user states. */
    if (c->ustates != NULL) {
        /* Free the queues. */
        for (uint32_t i = 0; i < c->n_users; i++) {
            mmap_free(c->ustates[i].head, c->qdepth * sizeof(request_t));
            io_uring_queue_exit(&c->ustates[i].ring);
        }

        /* Free the user states. */
        mmap_free(c->ustates, c->n_users * sizeof(ustate_t));
    }

    /* Kill the backend threads, if initialized. */
    KILL_NOT_ZERO(c->registrar_thread, SIGKILL);
    KILL_NOT_ZERO(c->monitor_thread, SIGKILL);
    KILL_NOT_ZERO(c->manager_thread, SIGKILL);

    /* Destroy the cache struct itself. */
    mmap_free(c, sizeof(cache_t));
}

/* Allocate a complete cache. Returns 0 on success and -errno on failure. */
int
cache_init(cache_t *c,
           size_t capacity,
           int debug_limit,
           int queue_depth,
           int max_unsynced,
           int n_users)
{
    /* Size arguments must all be >= 1, except for MAX_UNSYNCED, for which a
       zero value indicates infinite size. */
    assert(capacity >= 1);
    assert(queue_depth >= 1);
    assert(n_users >= 1);

    /* Allocate user states. */
    if ((c->ustates = mmap_alloc(n_users * sizeof(ustate_t))) == NULL) {
        LOG(LOG_CRITICAL, "mmap_alloc failed.\n");
        return -ENOMEM;
    }

    /* Initialize user states. */
    c->n_users = n_users;
    c->qdepth = queue_depth;
    for (uint32_t i = 0; i < c->n_users; i++) {
        ustate_t *ustate = &c->ustates[i];

        /* Allocate requests (queue entries). */
        if ((ustate->head = mmap_alloc(queue_depth * sizeof(request_t))) == NULL) {
            cache_destroy(c);
            LOG(LOG_CRITICAL, "mmap_alloc failed.\n");
            return -ENOMEM;
        }
        memset(ustate->head, 0, queue_depth * sizeof(request_t));
        ustate->free = ustate->head;

        /* Initialize requests. */
        for (int j = 0; j < queue_depth; j++) {
            request_t *queue = ustate->free;
            queue[j].next = j + 1 < queue_depth ? &queue[j + 1] : NULL;
            queue[j].prev = j - 1 < queue_depth ? &queue[j - 1] : NULL;
        }

        /* Easy access to the tail without creating a forward loop. */
        ustate->free[0].prev = &ustate->free[queue_depth - 1];

        /* The other queues start empty. */
        ustate->ready = NULL;
        ustate->done = NULL;
        ustate->cleanup = NULL;
        ustate->bottleneck = NULL;

        /* Initialize the io_uring queues. */
        int status = io_uring_queue_init(c->qdepth, &ustate->ring, 0);
        if (status < 0) {
            LOG(LOG_CRITICAL, "io_uring_queue_init failed.\n");
            cache_destroy(c);
            return status;
        }

        /* Queue statistics. */
        ustate->queue_depth = (size_t) queue_depth;
        ustate->debug_count = 0;
        ustate->debug_limit = debug_limit;
        atomic_init(&ustate->in_flight, 0);

        /* Initialize the locks. */
        SPIN_MUST_INIT(&ustate->free_lock);
        SPIN_MUST_INIT(&ustate->ready_lock);
        SPIN_MUST_INIT(&ustate->done_lock);
        SPIN_MUST_INIT(&ustate->cleanup_lock);
    }

    /* Set up the local cache. */
    c->lcache = (lcache_t) {
        .ht = NULL,
        .unsynced = NULL,
        .n_unsynced = 0,
        .threshold = max_unsynced,
        .capacity = capacity,
        .used = 0
    };

    /* Set up the remote cache. */
    c->rcache.ht = NULL;
    SPIN_MUST_INIT(&c->rcache.ht_lock);

    /* Set up the total cache. */
    c->registrar_thread = 0;
    c->monitor_thread = 0;
    c->manager_thread = 0;
    c->peers = NULL;
    SPIN_MUST_INIT(&c->peer_lock);

    /* Get a unique (actually random) number to serve as this machine's ID to
       avoid self-adding. See comments in cache.h for more details. */
    int rfd = open("/dev/urandom", O_RDONLY);
    if (rfd < 0) {
        LOG(LOG_CRITICAL, "Failed to open /dev/urandom; %s\n", strerror(errno));
        cache_destroy(c);
        return -errno;
    }
    ssize_t bytes = read(rfd, &c->random, sizeof(c->random));
    close(rfd);
    if (bytes != sizeof(c->random)) {
        LOG(LOG_CRITICAL, "Failed to read randomness from /dev/urandom.\n");
        cache_destroy(c);
        return -EBADFD;
    }

    return 0;
}
