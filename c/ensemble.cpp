//
//  main.cpp
//  yt8m_add_pred
//

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <fstream>
#include "pred.hpp"
#include "utils.hpp"
using namespace std;

int main(int argc, const char * argv[]) { 

    if (argc !=2 ) {
        cout<<"Usage:\n"<<argv[0]<<" file_wgt_list.csv\n";
        exit(1);
    }
    std::vector< std::pair<string, double> > files_and_wgt = read_files_and_wgts(argv[1]);

    if (files_and_wgt.size() < 4) {
        cout<<files_and_wgt.size()<<" files to be processed.\n";
        cout<<"Less than 4 predictions to process!\n";
        cout<<"Please use addPred or add3Pred instead."<<endl;
        exit(1);
    }
    
    std::vector<std::string> files;
    std::vector<double> wgts;
    for (auto x : files_and_wgt) {
         files.push_back(x.first);
         wgts.push_back(x.second);
    }
    
    double wgts_sum = 0;
    for (auto y : wgts)
        wgts_sum += y;
    for (auto it = wgts.begin(); it!= wgts.end(); it++)
        *it = *it/wgts_sum;
            
    std::cout<<currentDateTime()<<": Total "<<files.size()<<" files.\n";

    for (int i=0; i< files.size(); i++)
        cout<<files[i]<<" : wgt = "<<wgts[i]<<'\n';
    
    int split_at = files.size()/2;
       
    vector<prediction> c0, c1;
    std::thread u0 {[&] { 
        vector<prediction> x0 = file2preds(files[0]);
        vector<prediction> x1 = file2preds(files[1]);
        c0 = add_pred_vecs(x0, x1, wgts[0], wgts[1]);
        for (int i =2; i<split_at; i++) {
            x0 = file2preds(files[i]);
            c0 = add_pred_vecs(c0, x0, 1.0, wgts[i]);
        }
        std::cout<<currentDateTime()<<": Added "<<split_at <<'\n';
    }};

    std::thread u1 {[&] { 
        vector<prediction> x0 = file2preds(files[split_at]);
        vector<prediction> x1 = file2preds(files[split_at+1]);
        c1 = add_pred_vecs(x0, x1, wgts[split_at], wgts[split_at+1]);
        for (int i =split_at+2; i<files.size(); i++) {
            x0 = file2preds(files[i]);
            c1 = add_pred_vecs(c1, x0, 1.0, wgts[i]);
        }
        std::cout<<currentDateTime()<<": Added "<<files.size()-split_at<<'\n';
    }};

    u0.join();
    u1.join();

    vector<prediction> y0 = add_pred_vecs(c0, c1, 1.0, 1.0);
    std::cout<<currentDateTime()<<": Added "<<files.size()<<'\n';

    std::string fout = "/tmp/scrib_combo.csv";
    preds2file(y0, fout);
    std::cout<<currentDateTime()<<": Saved data to "<<fout<<'\n';
        
    return 0;
}
