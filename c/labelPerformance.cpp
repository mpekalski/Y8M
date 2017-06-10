//
//  labelPerformance.cpp 
//

#include <iostream>
#include <fstream>
#include <map>
#include "pred.hpp"
#include "utils.hpp"

int main(int argc, const char * argv[]) {
    if (argc < 3) {
        std::cout << "Usage:\n "<<argv[0]<<" true_label.csv prediction.csv [/tmp/label_perf_output.csv]\n";
        exit(1);
    }
    
    std::string fn1 = argv[1];
    std::string fn2 = argv[2];
    std::string fout = "/tmp/label_perf_output.csv";

    if (argc == 4)
	fout = argv[3];

    
    std::cout<<currentDateTime()<<": Starting...\n";

    std::vector<trueLabel> ts = file2labels(fn1);
    std::cout<<currentDateTime()<<": Read true label file: "<<fn1<<'\n';
    std::vector<prediction> ps = file2preds(fn2);
    std::cout<<currentDateTime()<<": Read prediction file: "<<fn2<<'\n';

    // label, count, cumu_prob
    std::map<int, std::pair<double, double> > label_probs = LabelProbs(ps, ts);

    std::ofstream f;
    f.open(fout);
    f << "label,count,cumulative_prob\n";
    for (auto x : label_probs)
	f << x.first<<','<<x.second.first<<','<<x.second.second<<'\n';
    f.close();
    std::cout<<currentDateTime()<<": Done. Saved to "<<fout<<'\n';
    return 0;
}
