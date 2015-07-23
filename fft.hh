#ifndef __fft_h
#define __fft_h

#include <clFFT/clFFT.h>
#include <vector>

#include "fftdata.hh"

class Fft {
    
public:
    Fft(size_t fft_size);

    bool init();
    
    void shutdown();

    size_t get_size() { return _fft_size; }
    
    bool add(FftData& data);

public:
    cl_context get_context() { return _context; }

private:
    bool setupCl();
    bool setupClFft();

private:
    size_t              _fft_size;

    cl_platform_id      _platform;
    cl_device_id        _device;
    cl_context          _context;
    cl_command_queue    _copy_queue;
    cl_command_queue    _fft_queue;

    clfftPlanHandle     _planHandle;
};

#endif // __fft_h