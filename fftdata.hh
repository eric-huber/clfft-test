#ifndef __fftdata_hh
#define __fftdata_hh

#include <clFFT/clFFT.h>

class Fft;

class FftData {

friend class Fft;

public:
    FftData(Fft& fft);
    ~FftData();

    void       get(cl_float* real, cl_float* imag);
    void       set(cl_float* real, cl_float* imag);
    void       wait();

    size_t     get_fft_size();

private:
    size_t     buffer_size();

private:
    enum        BufferType {REAL=0, IMAG=1};

private:
    Fft&        _fft;
    cl_float*   _real;
    cl_float*   _imag;
    cl_mem      _local_buffers[2];
    cl_event    _wait_list[2];
};

#endif // __fftdata_hh