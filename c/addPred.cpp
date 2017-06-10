//
//  main.cpp
//  yt8m_add_pred
//

#include <iostream>
#include <thread>
#include "pred.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[]) {
    std::string fn1, fn2, fn3 = "/tmp/combo.csv";
    double w1=0.5, w2=0.5;
    int topk=20;
    if (argc < 3) {
        std::cout<<"Usage:\n "<<argv[0]<<" in_file1 in_file2 [w1 w2 [out_file topk]]\n";
        std::cout<<"\n default values: w1 = 0.5  w2 = 0.5 out_file = \"/tmp/combo.csv\" topk = 20\n";
        exit(1);
    }
    
    fn1=argv[1];
    fn2=argv[2];
    
    if (argc >= 5) {
        w1 = std::stod(argv[3]);
        w2 = std::stod(argv[4]);
    }
    
    if (argc >= 6)
        fn3 = argv[5];
    if (argc == 7)
        topk = atoi(argv[6]);
    
    std::cout<<currentDateTime()<<": Starting...\n";
       
    std::vector<prediction> v1, v2;
    std::thread t1 {[&] { v1 = file2preds(fn1); }};
    std::thread t2 {[&] { v2 = file2preds(fn2); }};

    t1.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn1<<'\n';
    t2.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn2<<'\n';

    std::vector<prediction> v3 = add_pred_vecs(v1, v2, w1, w2, topk);
    std::cout<<currentDateTime()<<": Added two predictions with (w1, w2) = ("<<w1<<", "<<w2<<")\n";
    
    preds2file(v3, fn3, topk);
    std::cout<<currentDateTime()<<": Saved data to "<<fn3<<"\n";
    
    return 0;
}
