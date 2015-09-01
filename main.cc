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

void test_fft(size_t size, bool use_cpu, int parallel, long count, double range, double min) {

    Fft fft(size, use_cpu, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }
    
    FftJob job(size);
    job.randomize(range, min);

    job.dump("Initial random buffer");

    // perform fft
    fft.add(job);

    // wait for completion
    fft.wait_all();

    //buffer.is_finished();

    job.dump("FFT");

    fft.shutdown();
    // cleanup
}

void invert_fft(size_t size, bool use_cpu, int parallel, long count, double range, double min) {

    Fft fft(size, use_cpu, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }
    
    FftJob forward(size);
    //forward.randomize(range, min);
    forward.periodic();
    
    forward.write("fft-data.txt");
    
    // perform fft
    fft.add(forward);
    fft.wait_all();
    
    forward.write("fft-forward.txt");
    
    // buffer for inversion
    FftJob reverse(size);
    reverse.copy(forward);
    reverse.invert();
    reverse.write("fft-reverse-data.txt");
    
    // invert
    fft.add(reverse);
    fft.wait_all();
    
    reverse.invert();
    reverse.scale(1.0 / (double) size);
    
    reverse.write("fft-reverse.txt");
    
    forward.compare(reverse);
    
    fft.shutdown();
}

void time_fft(size_t size, bool use_cpu, int parallel, long count, double range, double min) {

    cout << "Timing..." << endl;

    Fft fft(size, use_cpu, parallel);
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
            fft.add(job);
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
    cout << "Hardware:   " << (use_cpu ? "CPU" : "GPU") << endl;
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

    size_t  fft_size    = 8192;
    bool    use_cpu     = false;
    int     parallel    = 16;
    long    count       = 1e9;
    double  range       = 25.0;
    double  min         = 0.0;
    bool    invert      = false;

    try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",      "Produce help message")
        ("cpu,c",       "Force CPU usage")
        ("parallel,p",  po::value<int>(), "Jobs to perform in parallel")
        ("iter,i",      po::value<long>(), "Set the number of iterations to perform")
        ("size,s",      po::value<int>(), "Set the size of the buffer buffer [8192]")
        ("range,r",     po::value<double>(), "Set the range of the random buffer [25.0]")
        ("min,m",       po::value<double>(), "Set the minimum value of the random buffer [0.0]")
        ("invert,i",    "Perform an FFT, then an inverse FFT on the same buffer");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }
        
        if (vm.count("cpu")) {
            use_cpu = true;
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
        
        if (vm.count("invert")) {
            invert = true;
        }

    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error" << endl;
        return 1;
    }

    // to nearest 16
    count = ((int) ceil(count / 16.0) + 1) * 16;

    //test_fft(fft_size, use_cpu, parallel, count, range, min);
    
    if (invert)
        invert_fft(fft_size, use_cpu, parallel, count, range, min);
    else    
        time_fft(fft_size, use_cpu, parallel, count, range, min);
    
    return 0;
}
