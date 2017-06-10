//
//  main.cpp
//  yt8m_add_pred
//

#include <iostream>
#include <thread>
#include "pred.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[]) {
   
    std::string fn1, fn2, fn3, fout = "/tmp/combo.csv";
    double w1=0.333, w2=0.334, w3 =0.333;
    if (argc < 4) {
        std::cout<<"Usage:\n "<<argv[0]<<" in_file1 in_file2 in_file3 [w1 w2 w3 [out_file]]\n";
        std::cout<<"\n default values: w1 = 0.333\n";
        std::cout<<"                 w2 = 0.334\n";
        std::cout<<"                 w3 = 0.333\n";
        std::cout<<"           out_file = \"tmp/combo.csv\"\n";
        exit(1);
    }
    
    fn1=argv[1];
    fn2=argv[2];
    fn3=argv[3];
    
    
    if (argc >= 7) {
        w1 = std::stod(argv[4]);
        w2 = std::stod(argv[5]);
        w3 = std::stod(argv[6]);
    }
    
    if (argc == 8)
        fout = argv[7];
    
    std::cout<<currentDateTime()<<": Starting...\n";
    
    std::vector<prediction> v1, v2, v3;
    std::thread t1 {[&]{ v1 = file2preds(fn1); } };
    std::thread t2 {[&]{ v2 = file2preds(fn2); } };
    std::thread t3 {[&]{ v3 = file2preds(fn3); } };

    t1.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn1<<'\n';
    t2.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn2<<'\n';
    t3.join();
    std::cout<<currentDateTime()<<": Read file: "<<fn3<<'\n';
    
    std::vector<prediction> c1 = add_pred_vecs(v1, v2, w1, w2);
    std::cout<<currentDateTime()<<": Added two predictions with (w1, w2) = ("<<w1<<", "<<w2<<")\n";
    std::vector<prediction> c2 = add_pred_vecs(c1, v3, 1.0, w3);
    std::cout<<currentDateTime()<<": Added the third prediction with w3 = "<<w3<<'\n';
    
    preds2file(c2, fout);
    std::cout<<currentDateTime()<<": Saved data to "<<fout<<"\n";
    
    return 0;
}
