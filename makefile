CC=g++
#CFLAGS+=-g
CXXFLAGS += -I /opt/intel/opencl/include
CXXFLAGS += -std=c++11
LDFLAGS  += -lboost_program_options
LDFLAGS  += -lclFFT -L/opt/intel/opencl -lm -lOpenCL 

PROG=clfft-test
OBJS=fft.o \
     fftjob.o \
     fftbuffer.o \
     main.o

.PHONY: all clean
$(PROG): $(OBJS)
	$(CC) -o $(PROG) $(OBJS) $(LDFLAGS)

%.o: %.cc
	$(CC) -c $(CXXFLAGS) $<

all: $(PROG)

clean:
	rm -f $(OBJS) $(PROG)