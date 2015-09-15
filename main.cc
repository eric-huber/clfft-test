#include <boost/program_options.hpp>

#include <clFFT.h>

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <iomanip>
 
#include "fft.hh"

using namespace std;
using namespace chrono;

namespace po = boost::program_options;

const char* _data_file_name = "fft-data.txt";
const char* _fft_file_name  = "fft-forward.txt";
const char* _bak_file_name  = "fft-backward.txt";

void test_fft(size_t size, Fft::Device device, int parallel, 
              long count, double range, double min) {

    Fft fft(size, device, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }
    
    FftJob job(size);
    job.randomize(range, min);

    job.write(_data_file_name);

    // perform fft
    fft.forward(job);

    // wait for completion
    fft.wait_all();

    //buffer.is_finished();

    job.write(_fft_file_name);

    fft.shutdown();
    // cleanup
}

void reverse_fft(size_t size, Fft::Device device, int parallel, 
                 long count, double range, double min) {

    Fft fft(size, device, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }
    
    FftJob forward(size);
    //forward.randomize(range, min);
    forward.periodic();
    
    forward.write(_data_file_name);
    
    // perform fft
    fft.forward(forward);
    fft.wait_all();
    
    forward.write(_fft_file_name);
    
    // buffer for inversion
    FftJob reverse(size);
    reverse.copy(forward);
    reverse.invert();
    
    // reverse
    fft.backward(reverse);
    fft.wait_all();
    
    reverse.invert();
    reverse.scale(1.0 / (double) size);
    
    reverse.write(_bak_file_name);
    
    forward.compare(reverse);
    
    fft.shutdown();
}

void time_fft(size_t size, Fft::Device device, int parallel, 
             long count, double range, double min) {

    cout << "Timing..." << endl;

    Fft fft(size, device, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }

    vector<FftJob> jobs;

    nanoseconds total_duration(0);
    int last_percent = -1;

    for (int outer = 0; outer < count; outer += parallel) {
        
        // randomize data 
        for (auto job : jobs) {
            job.randomize(range, min);
        }
        
        // start timer
        high_resolution_clock::time_point start = high_resolution_clock::now();
    
        // queue ffts
        for (auto job : jobs) {
            fft.forward(job);
        }
    
        // wait for completion
        fft.wait_all();

        // end timer            
        high_resolution_clock::time_point finish = high_resolution_clock::now();

        // compute time
        auto duration = finish - start;
        total_duration += duration_cast<nanoseconds>(duration);

        // update user
        int percent = (int) round((double) outer / (double) count * 100.0);
        if (percent != last_percent) {
            cerr << "\r" << percent << " %";
            cerr.flush();
            last_percent = percent;
        }
    }
    
    fft.shutdown();
    
    // report time
    double ave = total_duration.count() / count;

    cout.precision(8);
    cerr << "\r100 %" << endl;
    cout << endl;
    cout << "Hardware:   ";
    if (Fft::CPU == device)
        cout << "CPU" << endl;
    else
        cout << "GPU" << endl;
    cout << "Precision:  Single" << endl;
    cout << "Parallel:   " << parallel << endl;
    cout << "Iterations: " << count << endl;
    cout << "Data size:  " << size << endl;
    cout << "Range:      " << range << endl;
    cout << "Min:        " << min << endl;
    cout << endl;
    cout << "Time:       " << total_duration.count() << " ns" << endl;
    cout << "Average:    " << ave << " ns (" << (ave / 1000.0) << " μs)" << endl;  
}

int main(int ac, char* av[]) {

    size_t          fft_size        = 8192;
    Fft::Device     device          = Fft::GPU; 
    bool            reverse         = false;
    bool            time            = false;
    int             parallel        = 16;
    long            count           = 1e9;
    double          range           = 25.0;
    double          min             = 0.0;

    try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",      "Produce help message")
        ("cpu,c",       "Force CPU usage")
        ("reverse,r",   "Perform an FFT, then an inverse FFT on the same buffer")
        ("time,t",      "Time the FFT operation")
        ("parallel,p",  po::value<int>(), "Jobs to perform in parallel")
        ("iter,i",      po::value<long>(), "Set the number of iterations to perform")
        ("size,s",      po::value<int>(), "Set the size of the buffer buffer [8192]")
        ("range,a",     po::value<double>(), "Set the range of the random buffer [25.0]")
        ("min,m",       po::value<double>(), "Set the minimum value of the random buffer [0.0]");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }
        
        if (vm.count("cpu")) {
            device = Fft::CPU;
        }
        
        if (vm.count("reverse")) {
            reverse = true;
        }
        
        if (vm.count("time")) {
            time = true;
        }
        
        if (vm.count("parallel")) {
            parallel = vm["parallel"].as<int>();
        }

        if (vm.count("iter")) {
            count = vm["iter"].as<long>();
        }

        if (vm.count("size")) {
            fft_size = vm["size"].as<int>();
        }
        
        if (vm.count("range")) {
            range = vm["range"].as<double>();
        }
        
        if (vm.count("min")) {
            min = vm["min"].as<double>();
        }

    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error" << endl;
        return 1;
    }

    // to nearest 16
    count = ((int) ceil(count / parallel)) * parallel;

    if (reverse)
        reverse_fft(fft_size, device, parallel, count, range, min);
    else if (time)    
        time_fft(fft_size, device, parallel, count, range, min);
    else
        test_fft(fft_size, device, parallel, count, range, min);
    
    return 0;
}
