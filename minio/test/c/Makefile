# https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/

CC     = gcc
CFLAGS = -Wall -lpthread -lrt -g
DEPS   = ../../csrc/minio/minio.h ../../csrc/utils/utils.h ../../csrc/utils/uthash.h
OBJ    = test_minio.o ../../csrc/minio/minio.o ../../csrc/utils/utils.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

test: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f test $(OBJ)