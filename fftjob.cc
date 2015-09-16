#include "fftjob.hh"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

FftJob::FftJob(size_t size) 
 : _size(size)
{
    _data  = new cl_float[_size];
}


FftJob::~FftJob() {
    release();
}

void FftJob::copy(FftJob& other) {
    for (int i = 0; i < _size; ++i) {
        _data[i] = other._data[i];
    }    
}

double FftJob::rms(FftJob& inverse) {
    
    double rms = 0;
    
    for (int i = 0; i < _size; ++i) {
        rms += pow(inverse.at(i) - at(i), 2);
    }
    rms /= size();
    rms = sqrt(rms);
    return rms;
}

double FftJob::signal_to_quant_error(FftJob& inverse) {
    
    return 10.0 * log10( signal_energy() / quant_error_energy(inverse) );
}

double FftJob::signal_energy() {
    double energy = 0;
    for (int i = 0; i < _size; ++i) {
        energy += pow(at(i), 2);
    }
    return energy;
}

double FftJob::quant_error_energy(FftJob& inverse) {
    
    double energy = 0;
    for (int i = 0; i < _size; ++i) {
        energy += pow(at(i) - inverse.at(i), 2);
    }
    return energy;
}

void FftJob::populate(FftJob::TestData data_type) {
    switch (data_type) {
    case PERIODIC:
    default:
        periodic();
        break;
    case RANDOM:
        randomize(2, -1);
        break;
    }
}

void FftJob::randomize(double range, double min) {
    
    srand(time(NULL));

    for(int i = 0; i < _size; i++) {
        double rnd = (float) rand() / RAND_MAX * range + min;
        _data[i]  = rnd;
    }
}

void FftJob::periodic() {
    for (int i = 0; i < _size; ++i) {
        double t = i * .002;
        double amp = sin(2 * M_PI * t); 
        _data[i] = amp;
    }
}

void FftJob::scale(double factor) {
    for (int i = 0; i < _size; ++i) {
        _data[i] *= factor;
    }
}

void FftJob::dump(std::string label) {

    std::cout << label << std::endl;
    
    for (int i = 0; i < 32; i = i + 4) {
        for (int j = 0; j < 4; ++j) { 
            std::cout << std::setw(5) << std::setprecision(1) << std::fixed << std::setfill(' ') << _data[i + j];
            if (3 != j)
                std::cout << "    ";
        }
        std::cout << std::endl;
    }
}

void FftJob::write(std::string filename) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 0; i < _size; ++i) {
        ofs << _data[i] << std::endl;
    }
    
    ofs.close();   
}

void FftJob::write_hermitian(std::string filename) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 0; i < _size / 2; ++i) {
        auto real = at_hr(i);
        auto imag = at_hi(i);
        auto amplitude = sqrt(pow(real, 2) + pow(imag, 2));
        auto phase = atan2(imag, real);
        ofs << real << ", " << imag << ", " << amplitude << ", " << phase << std::endl;
    }
    
    ofs.close();   
}

void FftJob::release() {
    if (NULL != _data) {
        delete[] _data;
        _data = NULL;
    }
}