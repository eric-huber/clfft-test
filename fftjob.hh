#include <clFFT.h>
#include <string>

class FftJob {

public:
    FftJob(size_t fft_size);
    ~FftJob();
    
public:
    void        randomize(double range, double min);
    void        dump(std::string label);
    void        release();
  
    cl_float*   real() { return _real; }
    cl_float*   imag() { return _imag; }  
  
private:
    size_t      _size;

    cl_float*   _real;
    cl_float*   _imag;    
};