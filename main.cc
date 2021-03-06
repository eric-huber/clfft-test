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

void test_fft(size_t size, Fft::Device device, FftJob::TestData test_data, 
		      int parallel, long count, double mean, double std) {

    Fft fft(size, device, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }
    
    FftJob job(size, mean, std);
    job.populate(test_data);
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

void inverse_fft(size_t size, Fft::Device device,  FftJob::TestData test_data, 
	             int parallel, long count, double mean, double std) {

    Fft fft(size, device, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }
    
    FftJob data(size, mean, std);
    data.populate(test_data);
    
    data.write(_data_file_name);
    
    // perform fft
    FftJob forward(size, mean, std);
    forward.copy(data); // we need to preserve the original data - in place clobbers it
    fft.forward(forward);
    fft.wait_all();
    
    forward.write_hermitian(_fft_file_name);
    
    // buffer for inversion
    FftJob reverse(size, mean, std);
    reverse.copy(forward);
    
    // reverse
    fft.backward(reverse);
    fft.wait_all();
   
    reverse.write(_bak_file_name);
    
    cout << "FFT/IFFT computed." << endl;
    cout << "Data saved." << endl;
    cout << "Root Mean Square :              " << std::setprecision(4) 
        << data.rms(reverse) << endl;
    cout << "Signal to Quantinization Error: " << std::setprecision(4) 
        << data.signal_to_quant_error(reverse) << endl;
    
    fft.shutdown();
}

void inverse_fft_loop(size_t size, Fft::Device device,  FftJob::TestData test_data, 
	                  int parallel, long count, double mean, double std) {


    Fft fft(size, device, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }
    
    double sqer = 0;
    int last_percent = -1;
    
    for (int l = 0; l < count; ++l) {
    
        FftJob data(size, mean, std);
        data.populate(test_data);
        
        // perform fft
        FftJob forward(size, mean, std);
        forward.copy(data); // we need to preserve the original data - in place clobbers it
        fft.forward(forward);
        fft.wait_all();
        
        // buffer for inversion
        FftJob reverse(size, mean, std);
        reverse.copy(forward);
        
        // reverse
        fft.backward(reverse);
        fft.wait_all();

        sqer += data.signal_to_quant_error(reverse);
        
        // update user
        int percent = (int) round((double) l / (double) count * 100.0);
        if (percent != last_percent) {
            cerr << "\r" << percent << " %";
            cerr.flush();
            last_percent = percent;
        } 
    }
        
    sqer /= (double) count;
    
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
    cout << "Data type:  ";
    if (FftJob::PERIODIC) {
        cout << "Periodic" << endl;
    } else {
        cout << "Random" << endl;
        cout << "Mean:      " << mean << endl;
        cout << "Std:       " << std << endl;
    }   
    cout << "Ave Signal to Quantinization Error: " << std::setprecision(4) 
        << sqer << endl;
    
    fft.shutdown();
}

void time_fft(size_t size, Fft::Device device, FftJob::TestData test_data, 
	      int parallel, long count, double mean, double std) {

    cout << "Timing..." << endl;

    Fft fft(size, device, parallel);
    if (!fft.init()) {
        fft.shutdown();
        return;
    }

    vector<FftJob*> jobs;
    for (int i = 0; i < parallel; ++i) {
        jobs.push_back(new FftJob(size, mean, std));
    }
    
    nanoseconds total_duration(0);
    int last_percent = -1;

    for (int outer = 0; outer < count; outer += parallel) {
        
        // randomize data
        for (auto job : jobs) {
            job->populate(test_data);
        }
            
        // start timer
        high_resolution_clock::time_point start = high_resolution_clock::now();
    
        // queue ffts
        for (auto job : jobs) {
            fft.forward(*job);
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
    cout << "Data type:  ";
    if (FftJob::PERIODIC) {
        cout << "Periodic" << endl;
    } else {
        cout << "Random" << endl;
        cout << "Mean:      " << mean << endl;
        cout << "Std:       " << std << endl;
    }
    cout << endl;
    cout << "Time:       " << total_duration.count() << " ns" << endl;
    cout << "Average:    " << ave << " ns (" << (ave / 1000.0) << " μs)" << endl;  
}

int main(int ac, char* av[]) {

    size_t              fft_size        = 8192;
    Fft::Device         device          = Fft::GPU;
    FftJob::TestData    test_data       = FftJob::RANDOM;
    bool                inverse         = false;
    bool                inverse_loop    = false;
    bool                time            = false;
    int                 parallel        = 16;
    long                count           = 1000;
    double              mean            = 0.5;
    double              std             = 0.2;

    try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",         "Produce help message")
        ("cpu,c",          "Force CPU usage")

        ("inverse,i",      "Perform an FFT, then an inverse FFT on the same buffer")
        ("inverse-loop,v", "Compute average SQER")
        ("time,t",         "Time the FFT operation")
        
        ("periodic,p",     "Use a periodic data set")
        ("random,r",       "Use a gaussian distributed random data set")
        ("mean,m",         po::value<double>(), "Mean for random data")
        ("deviation,d",    po::value<double>(), "Standard deviation for random data")
        
        ("jobs,j",         po::value<int>(), "Jobs to perform in parallel")
        ("loops,l",        po::value<long>(), "Set the number of iterations to perform")
        ("size,s",         po::value<int>(), "Set the size of the buffer [8192]");

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
        
        if (vm.count("inverse")) {
            inverse = true;
        }
        
        if (vm.count("inverse-loop")) {
            inverse_loop = true;
        }
        
        if (vm.count("time")) {
            time = true;
        }
        
        if (vm.count("periodic")) {
        	test_data = FftJob::PERIODIC;
        }
        
        if (vm.count("random")) {
        	test_data = FftJob::RANDOM;
        }
        
        if (vm.count("mean")) {
            mean = vm["mean"].as<double>();
        }

        if (vm.count("deviation")) {
            std = vm["deviation"].as<double>();
        }
        
        if (vm.count("jobs")) {
            parallel = vm["jobs"].as<int>();
        }

        if (vm.count("loops")) {
            count = vm["loops"].as<long>();
        }

        if (vm.count("size")) {
            fft_size = vm["size"].as<int>();
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

    if (inverse)
        inverse_fft(fft_size, device, test_data, parallel, count, mean, std);
    else if (inverse_loop)
        inverse_fft_loop(fft_size, device, test_data, parallel, count, mean, std);
    else if (time)    
        time_fft(fft_size, device, test_data, parallel, count, mean, std);
    else
        test_fft(fft_size, device, test_data, parallel, count, mean, std);
    
    return 0;
}
