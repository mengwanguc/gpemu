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

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include "../../../csrc/utils/alloc.h"
#include "../../../csrc/async/async.h"


/* Generic worker process. */
void
test_worker_loop(wstate_t *worker,
                 uint64_t id,
                 char **filepaths,
                 size_t n_filepaths)
{
    struct timespec start, request_end, retrieve_end, release_end;
    entry_t *entries[n_filepaths];

    /* Request all files to loader. */
    clock_gettime(CLOCK_REALTIME, &start);
    for (size_t i = 0; i < n_filepaths; i++) {
        while (!async_try_request(worker, filepaths[i])) {}
    }
    clock_gettime(CLOCK_REALTIME, &request_end);

    /* Retrieve all files from loader. */
    for (size_t i = 0; i < n_filepaths; i++) {
        while ((entries[i] = async_try_get(worker)) == NULL) {}
    }
    clock_gettime(CLOCK_REALTIME, &retrieve_end);

    /* Release all entries. */
    for (size_t i = 0; i < n_filepaths; i++) {
        async_release(entries[i]);
    }
    clock_gettime(CLOCK_REALTIME, &release_end);


    /* Log timing data. */
    long request_time = request_end.tv_nsec - start.tv_nsec + (request_end.tv_sec - start.tv_sec) * 1e9;
    long retrieve_time = retrieve_end.tv_nsec - start.tv_nsec + (retrieve_end.tv_sec - start.tv_sec) * 1e9;
    long release_time = release_end.tv_nsec - start.tv_nsec + (release_end.tv_sec - start.tv_sec) * 1e9;

    printf("Worker %lu results --\n"
           "\t Request time: %ld ns\n"
           "\tRetrieve time: %ld ns (delta %ld ns)\n"
           "\t Release time: %ld ns (delta %ld ns)\n",
           id,
           request_time,
           retrieve_time, retrieve_time - request_time,
           release_time, release_time - retrieve_time);
}


void
test_config(size_t queue_depth,
            size_t n_workers,
            size_t dispatch_n,
            size_t idle_iters,
            char **filepaths,
            size_t n_filepaths)
{
    printf("\n-- Testing config with %lu worker(s) --\n", n_workers);

    /* Create the loader. */
    lstate_t *loader = mmap_alloc(sizeof(lstate_t));
    assert(loader != NULL);

    /* Initialize the loader. */
    int status = async_init(loader, queue_depth, n_workers, dispatch_n, idle_iters, 0);
    assert(status == 0);

    /* Fork, spawning worker processes. */
    pid_t worker_pids[n_workers];
    size_t fp_per_worker = n_filepaths / n_workers;
    for (size_t i = 0; i < n_workers; i++) {
        if ((worker_pids[i] = fork()) > 0) {
            continue;
        }

        /* Start child as a worker. */
        test_worker_loop(&loader->states[i], i, filepaths + fp_per_worker * i, fp_per_worker);

        /* Exit worker upon completion. */
        printf("Worker %lu exiting.\n", i);
        exit(EXIT_SUCCESS);
    }

    /* Fork, spawning loader process. */
    pid_t loader_pid;
    if ((loader_pid = fork()) == 0) {
        /* Child becomes loader process. Should never return. */
        async_start(loader);

        /* If the loader returned, something went wrong. */
        printf("!!! loader returned !!!\n");
        exit(EXIT_FAILURE);
    }

    /* Wait on the worker processes. */
    for (size_t i = 0; i < n_workers; i++) {
        int status;
        waitpid(worker_pids[i], &status, 0);
        assert(status == EXIT_SUCCESS);
    }

    /* Kill the loader process. */
    printf("All workers have terminated. Killing loader.\n");
    kill(loader_pid, SIGKILL);
}

int
main(int argc, char **argv)
{
    size_t n_filepaths    = 4;
    size_t queue_depth    = n_filepaths;
    size_t dispatch_n = queue_depth;
    size_t idle_iters = 64;
    char *filepaths[] = {
        "Makefile",
        "async",
        "test_async.c",
        "test_async.o",
    };

    /* Worker configs to test. */
    size_t n_workers[] = {1, 2};
    size_t n_configs = 2;

    /* Run each test configuration. */
    for (size_t i = 0; i < n_configs; i++) {
        test_config(queue_depth,
                    n_workers[i],
                    dispatch_n,
                    idle_iters,
                    filepaths,
                    n_filepaths);
    }

    printf("All tests complete.\n");

    return EXIT_SUCCESS;
}