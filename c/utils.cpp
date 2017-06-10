//
//  utils.cpp
//  yt8m utils
//

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <vector>

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    
    return buf;
}

std::vector< std::pair<std::string, double> > 
read_files_and_wgts(const std::string & files_list_csv) {
    // files_list_csv is the filename that contains this format:
    //   file_path_1,wgt1
    //   file_path_2,wgt2
    //   ...
    // No header

    std::ifstream f;
    f.open(files_list_csv);

    std::vector< std::pair<std::string, double> > res;

    std::string line, fname, wgt_str;
    bool file_error=false;

    while(getline(f, line)) {
        if (line[0] == '#' || line[0] == ' ')
           continue;
        std::stringstream ss(line);
        getline(ss, fname, ',');
        getline(ss, wgt_str);
        //std::cout<<"Checking "<<fname<<'\n';
        if (! std::ifstream(fname).good()) {
            std::cout<<"File error: "<<fname<<'\n';
            file_error = true;
        }
        res.push_back( std::make_pair<std::string, double> (fname.c_str(), stod(wgt_str) ));
    }
    if (file_error)
        exit(1);

    return res;
}
