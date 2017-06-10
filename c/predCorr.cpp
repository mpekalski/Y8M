//
// predCorr.cpp
// yt8m correclation of two predictions.
//

#include <iostream>
#include <string>
#include <vector>
#include <thread>

#include "pred.hpp"
#include "corr.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[]) {
    std::string fn1, fn2;
    if (argc < 3) {
        std::cout<<"Usage:\n "<<argv[0]<<" in_file1 in_file2\n";
        exit(1);
    }
    
    fn1=argv[1];
    fn2=argv[2];

    std::cout<<currentDateTime()<<": Starting...\n";
       
    std::vector<prediction> v1, v2;
    std::thread t1 {[&] { v1 = file2preds(fn1); }};
    std::thread t2 {[&] { v2 = file2preds(fn2); }};

    t1.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn1<<'\n';
    t2.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn2<<'\n';

    double cor = pred_vecs_corr(v1, v2);
    double cov = pred_vecs_corr(v1, v2, 1.0/4716, false);
    double var1 = pred_vec_var(v1);
    double var2 = pred_vec_var(v2);
    std::cout<<currentDateTime()<<": correlation = "<<cor;
    std::cout<<", covariance = "<<cov<<", var(v1) = "<<var1<<", var(v2) = "<<var2<<std::endl;
    return 0;
}


