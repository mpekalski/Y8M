//
//  main.cpp
//  yt8m_add_pred
//

#include <iostream>
#include "pred.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[]) { 
    std::string fn2 = "/tmp/predictions2.csv";
    std::string fn3 = "/tmp/predictions3.csv";
    std::string fn4 = "/tmp/predictions4.csv";
    std::string fn5 = "/tmp/predictions5.csv";
    std::string fn6 = "/tmp/predictions6.csv";
        
    std::vector<prediction> v2 = file2preds(fn2);
    std::cout<<currentDateTime()<<": Read file: "<<fn2<<'\n';
    std::vector<prediction> v3 = file2preds(fn3);
    std::cout<<currentDateTime()<<": Read file: "<<fn3<<'\n';
    std::vector<prediction> v4 = file2preds(fn4);
    std::cout<<currentDateTime()<<": Read file: "<<fn4<<'\n';
    std::vector<prediction> v5 = file2preds(fn5);
    std::cout<<currentDateTime()<<": Read file: "<<fn5<<'\n';
    std::vector<prediction> v6 = file2preds(fn6);
    std::cout<<currentDateTime()<<": Read file: "<<fn6<<'\n';
    
    std::vector<prediction> c1  = add_pred_vecs(v2, v3, 0.2, 0.2, 40);
    std::cout<<currentDateTime()<<": Added v2 v3\n";
    std::vector<prediction> c2  = add_pred_vecs(c1, v4, 1.0, 0.2, 40);
    std::cout<<currentDateTime()<<": Added v4\n";
    std::vector<prediction> c3  = add_pred_vecs(c2, v5, 1.0, 0.2, 40);
    std::cout<<currentDateTime()<<": Added v5\n";
    std::vector<prediction> c4  = add_pred_vecs(c3, v6, 1.0, 0.2);
    std::cout<<currentDateTime()<<": Added v6\n";
    
    std::string fout = "../submissions/pred.Marcin.earlier.MoE4.csv";
    preds2file(c4, fout);
    std::cout<<currentDateTime()<<": Saved data to "<<fout<<'\n';
        
    return 0;
}
