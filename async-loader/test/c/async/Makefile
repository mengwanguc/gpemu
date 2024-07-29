CC     = gcc
CFLAGS = -Wall -lpthread -luring -lrt -g
DEPS   = ../../../csrc/async/async.h ../../../csrc/utils/alloc.h ../../../csrc/utils/sort.h
OBJ    = test_async.o ../../../csrc/async/async.o ../../../csrc/utils/alloc.o ../../../csrc/utils/sort.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

async: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f async $(OBJ)