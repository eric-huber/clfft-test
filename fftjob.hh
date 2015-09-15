#include <clFFT.h>
#include <string>

class FftJob {

public:
    FftJob(size_t fft_size);
    ~FftJob();
    
public:
    void        copy(FftJob& other);
    
    double      compare(FftJob& other);
    
    double      signal_to_quant_error(FftJob& inverse);
    double      signal_energy();
    double      quant_error_energy(FftJob& inverse);
    

    void        randomize(double range, double min);
    void        periodic();

    void        invert();
    void        scale(double factor);

    void        dump(std::string label);
    void        write(std::string file);

    void        release();
  
    cl_float*   real() { return _real; }
    cl_float*   imag() { return _imag; }

  
private:
    size_t      _size;

    cl_float*   _real;
    cl_float*   _imag;
};