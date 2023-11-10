CC = nvcc

all : vector_add.x

main.o: main.cu
	$(CC) -c main.cu -o main.o

vector.o: vector.cu
	$(CC) -c vector.cu -o vector.o

vector_add.x: main.o get_time.o vector.o
	$(CC) -o $@ $^

clean :
	\rm -f *.o vector_add.x





