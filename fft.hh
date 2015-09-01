#ifndef __fft_h
#define __fft_h

#include <clFFT.h>
#include <vector>

#include "fftjob.hh"
#include "fftbuffer.hh"

class Fft {
    
public:
    Fft(size_t fft_size, bool use_cpu, int parallel);

    bool    init();    
    void    shutdown();

    size_t  get_size() { return _fft_size; }
    
    bool    add(FftJob& job);
    void    wait_all();

public:
    cl_context  get_context() { return _context; }
    size_t      get_temp_buffer_size();

private:
    bool setup_cl();
    bool setup_clFft();
    bool setup_buffers();

    FftBuffer*  get_buffer();

private:
    size_t                  _fft_size;
    bool                    _use_cpu;
    int                     _parallel;

    cl_platform_id          _platform;
    cl_device_id            _device;
    cl_context              _context;
    cl_command_queue        _queue;
    clfftPlanHandle         _planHandle;
    
    std::vector<FftBuffer*> _buffers;
};

#endif // __fft_h
