#include <clFFT.h>
#include <string>

class FftJob {
public:
    enum TestData  {PERIODIC, RANDOM};
    
public:
    FftJob(size_t fft_size, double mean, double std);
    ~FftJob();
    
public:
    void        copy(FftJob& other);
    
    double      rms(FftJob& inverse);    
    double      signal_to_quant_error(FftJob& inverse);
    double      signal_energy();
    double      quant_error_energy(FftJob& inverse);  

    void        populate(TestData data_type);

    void        scale(double factor);

    void        dump(std::string label);
    void        write(std::string file);
    void        write_hermitian(std::string file);

    void        release();
  
    cl_float*   data()              { return _data; }

    cl_float    at(int index)       { return _data[index]; }
    cl_float    at_hr(int index)    { return _data[2 * index]; }
    cl_float    at_hi(int index)    { return _data[2 * index + 1]; }
    
    int         size()              { return _size; }
    int         size_h()            { return _size / 2; }

private:
    void        randomize();
    void        periodic();
    
private:
    size_t      _size;
    double      _mean;
    double      _std;

    cl_float*   _data;
};