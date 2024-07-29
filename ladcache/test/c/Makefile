# MIT License

# Copyright (c) 2023 Gus Waldspurger

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

CC     = gcc
CFLAGS = -Wall -Wno-unused-function -Wno-unused-variable -lpthread -luring -lrt -g -fstack-protector-all -Wstack-protector --param ssp-buffer-size=4
DEPS   = ../../csrc/ladcache/cache.h ../../csrc/utils/alloc.h ../../csrc/utils/fifo.h ../../csrc/log.h ../../csrc/uthash.h
OBJ    = test_ladcache.o ../../csrc/ladcache/cache.o ../../csrc/utils/alloc.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

ladcache: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f ladcache $(OBJ)