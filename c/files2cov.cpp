//
//  main.cpp
//  calculate the covariance matrix of predictions
//

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <fstream>
#include "pred.hpp"
#include "corr.hpp"
#include "utils.hpp"
using namespace std;

int main(int argc, const char * argv[]) { 

    if (argc !=2 ) {
        cout<<"Usage:\n"<<argv[0]<<" file_wgt_list.csv\n";
        exit(1);
    }
    std::vector<std::string> files;
    std::vector< std::pair<string, double> > files_and_wgt = read_files_and_wgts(argv[1]);
    
    for (auto x : files_and_wgt)
         files.push_back(x.first);

    std::cout<<currentDateTime()<<": Total "<<files.size()<<" files.\n";

    std::cout<<currentDateTime()<<": Starting...\n";
    
    vector<vector<prediction>> pred_vecs;
    
    for (auto f : files) {
        pred_vecs.push_back(file2preds(f));
        std::cout<<currentDateTime()<<": read "<<f<<'\n';
    }
    
    int n=0;
    std::cout<<currentDateTime()<<": covariance matrix:\n";
    for (auto it = pred_vecs.begin(); it != pred_vecs.end(); it++) {
        for (int k=0; k<n; k++)
            cout<<",";
        cout<< pred_vec_var(*it);
        for (auto it2 = it + 1; it2 != pred_vecs.end(); it2 ++)
            cout<<","<<pred_vecs_covar(*it, *it2);
        cout<<'\n';
        n++;
    }

    n = 0;
    std::cout<<currentDateTime()<<": correlation matrix:\n";
    for (auto it = pred_vecs.begin(); it != pred_vecs.end(); it++) {
        for (int k=0; k<n; k++)
            cout<<",";
        cout<< 1.0;
        for (auto it2 = it + 1; it2 != pred_vecs.end(); it2 ++)
            cout<<","<<pred_vecs_corr(*it, *it2);
        cout<<'\n';
        n++;
    }

    std::cout<<currentDateTime()<<": Done.\n";
    return 0;
}
