//
//  main.cpp
//  calc_GAP
//

#include <iostream>
#include <thread>
#include "pred.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[]) {
    if (argc != 3) {
        std::cout << "Usage:\n "<<argv[0]<<" true_label.csv prediction.csv\n";
        exit(1);
    }
    
    std::string fn1 = argv[1];
    std::string fn2 = argv[2];
    
    std::cout<<currentDateTime()<<": Starting...\n";

    std::vector<trueLabel> ts;
    std::thread t1 {[&] {ts = file2labels(fn1);}};
    std::vector<prediction> ps;
    std::thread t2 {[&] {ps = file2preds(fn2);}};

    t1.join();
    std::cout<<currentDateTime()<<": Read true labels file: "<<fn1<<'\n';
    t2.join();
    std::cout<<currentDateTime()<<": Read predictions file: "<<fn2<<'\n';

    double gap = GAP(ps, ts);
    std::cout<<currentDateTime()<<": Done.\n";
    std::cout<<"GAP = "<<gap<<'\n';

    return 0;
}
