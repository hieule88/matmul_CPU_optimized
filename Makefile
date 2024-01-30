TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++11 -O3 -Wall -march=znver2 -mavx2 -mfma -fopenmp
LDLIBS=-lm -lmpi -lmpi_cxx -pthread -lnuma

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)