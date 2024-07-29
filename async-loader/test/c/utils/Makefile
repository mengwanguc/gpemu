CC     = gcc
CFLAGS = -Wall -g
DEPS   = ../../../csrc/utils/sort.h
OBJ    = test_utils.o ../../../csrc/utils/sort.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

utils: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f utils $(OBJ)