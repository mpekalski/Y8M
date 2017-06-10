//
//  utils.hpp
//  yt8m utils
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <string>
#include <map>
#include <vector>


const std::string currentDateTime();
std::vector< std::pair<std::string, double> > read_files_and_wgts(const std::string & files_list_csv);
#endif


