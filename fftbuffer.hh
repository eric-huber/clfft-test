#ifndef __FftBuffer_hh
#define __FftBuffer_hh

#include <clFFT.h>

class Fft;

class FftBuffer {

friend class Fft;

public:
    FftBuffer(Fft& fft);
    ~FftBuffer();

    void        set_job(FftJob* job)        { _job = job; }
    FftJob*     get_job()                   { return _job; }

    void        wait();
    bool        is_finished();

    void        release();

    size_t      get_fft_size();

    bool        in_use()                    { return _in_use; }
    void        set_in_use(bool in_use)     { _in_use = in_use; }

private:
    size_t      size();

private:
    cl_float*   job_data()                  { return _job->data(); }

    cl_mem      data()                      { return _data_buf; }
    cl_mem*     data_addr()                 { return &_data_buf; }
    cl_mem      temp()                      { return _temp_buf; }

    void        set_wait(cl_event wait)     { _wait = wait; }

private:
    Fft&        _fft;
    FftJob*     _job;
    
    cl_mem      _data_buf;
    cl_mem      _temp_buf;
    
    cl_event    _wait;
    
    bool        _in_use;
};

#endif // __FftBuffer_hh
