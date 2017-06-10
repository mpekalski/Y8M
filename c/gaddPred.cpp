//
//  main.cpp
//  yt8m_add_pred
//

#include <iostream>
#include <thread>
#include "pred.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[]) {
    std::string fn1, fn2, fno = "/tmp/gcombo.csv";
    int topk=20;
    if (argc < 3) {
        std::cout<<"Usage:\n "<<argv[0]<<" in_file1 in_file2\n";
        std::cout<<"\n geometric average, weights are equal.\n";
        exit(1);
    }
    
    fn1=argv[1];
    fn2=argv[2];
    
    if (argc >= 4)
        fno = argv[3];
    if (argc == 5)
        topk = atoi(argv[4]);
    
    std::cout<<currentDateTime()<<": Starting...\n";
       
    std::vector<prediction> v1, v2;
    std::thread t1 {[&] { v1 = file2preds(fn1); }};
    std::thread t2 {[&] { v2 = file2preds(fn2); }};
    t1.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn1<<'\n';
    t2.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn2<<'\n';

    std::vector<prediction> v0 = gadd_pred_vecs(v1, v2);
    std::cout<<currentDateTime()<<": Geometrically added three predictions.\n";
    
    preds2file(v0, fno, topk);
    std::cout<<currentDateTime()<<": Saved data to "<<fno<<"\n";
    
    return 0;
}
