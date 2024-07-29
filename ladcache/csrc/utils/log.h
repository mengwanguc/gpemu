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

#ifndef __UTILS_LOG_H_
#define __UTILS_LOG_H_

#include <unistd.h>
#include <time.h>

/* Log scopes. */
enum log_scope {
    SCOPE_INT,
    SCOPE_EXT
};

/* Logging scope prefixes, indexed by log scopes enum. */
static const char *scope_prefixes[] = {
    "INT",
    "EXT"
};

/* Log levels. */
enum log_level {
    LOG_CRITICAL,
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
    LOG_DEBUG,
    LOG_ALL
};

/* Logging level prefixes, indexed by log levels enum. */
static const char *level_prefixes[] = {
    "CRIT",
    "ERR.",
    "WARN",
    "INFO",
    "DBG."
};

/* Change this to modify logging settings. */
#define LOG_MIN_LEVEL LOG_INFO

/* Master on/off setting. */
#define DEBUG 1

#define CONCAT(a, b) a ## b
#define DEBUG_LOG(scope, level, fmt, ...)                                      \
    do {                                                                       \
        time_t __log_time;                                                     \
        time(&__log_time);                                                     \
        struct tm *__log_tm = localtime(&__log_time);                          \
        if (DEBUG && level <= LOG_MIN_LEVEL) {                                 \
            fprintf(stderr,                                                    \
                    "[PID %d][%d:%d:%d][%s][%s][%s:%d] " fmt,                  \
                    getpid(),                                                  \
                    __log_tm->tm_hour,                                         \
                    __log_tm->tm_min,                                          \
                    __log_tm->tm_sec,                                          \
                    scope_prefixes[scope],                                     \
                    level_prefixes[level],                                     \
                    __FILE__,                                                  \
                    __LINE__,                                                  \
                    ## __VA_ARGS__);                                           \
        }                                                                      \
    } while (0)

#endif
