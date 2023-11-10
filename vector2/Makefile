CC = nvcc
DEPS = vector.h timer.h

all : vector_add.x

vector_add.x: main.cu
	$(CC) -o $@ $^

clean :
	\rm -f vector_add.x





