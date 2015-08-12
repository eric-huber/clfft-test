#ifndef __FftBuffer_hh
#define __FftBuffer_hh

#include <clFFT.h>

class Fft;

class FftBuffer {

friend class Fft;

public:
    FftBuffer(Fft& fft);
    ~FftBuffer();

    void        set_job(FftJob* job) { _job = job; }
    FftJob*     get_job() { return _job; }

    void        wait();
    bool        is_finished();

    void        release();

    size_t      get_fft_size();

private:
    size_t      buffer_size();

private:
    enum        BufferType {REAL=0, IMAG=1};

    cl_float*   data_real()     { return _job->real(); }
    cl_float*   data_imag()     { return _job->imag(); }

    cl_mem*     in_buffers()    { return _in_buf; }
    cl_mem*     out_buffers()   { return _out_buf; }
    cl_mem      temp_buffer()   { return _temp_buf; }
    
    cl_mem      in_real()       { return _in_buf[REAL]; }
    cl_mem      in_imag()       { return _in_buf[IMAG]; }

    cl_mem      out_real()      { return _out_buf[REAL]; }
    cl_mem      out_imag()      { return _out_buf[IMAG]; }

    void        set_wait(cl_event* wait) { 
                        _wait_list[REAL] = wait[REAL]; 
                        _wait_list[IMAG] = wait[IMAG];
                    }

private:
    Fft&        _fft;
    FftJob*     _job;
    
    cl_mem      _in_buf[2];
    cl_mem      _out_buf[2];
    cl_mem      _temp_buf;
    
    cl_event    _wait_list[2];
};

#endif // __FftBuffer_hh
