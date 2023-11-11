CC = nvcc
DEPS = vector.h timer.h

all : vector_add.x

vector_add.x: main.cu
	$(CC) -std=c++14 -O3 -o $@ $^

clean :
	\rm -f vector_add.x





