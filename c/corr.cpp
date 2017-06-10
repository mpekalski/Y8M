//
//  corr.hpp
//  yt8m corr of predictions.
//

#include <iostream>
#include <cmath>
#include <thread>
#include <algorithm>
#include "corr.hpp"

double pred_var(const prediction& p, double base) {
    double v=0;
    for (auto it=p.probs.begin(); it!=p.probs.end(); it++)
	v += (it->second-base) * (it->second - base);
    return v;
}

double preds_dot_prod(prediction& p1, prediction& p2, double base) {
    if (p1.vid.compare(p2.vid) != 0) {
        std::cout<<"incompatible video ids: "<<p1.vid<<", "<<p2.vid<<'\n';
        exit(1);
    }
 
    double v=0;
    for (auto it=p1.probs.begin(); it!= p1.probs.end(); it++)
	v += (it->second - base) * ( p2.probs[it->first] - base);
    return v;
}

double preds_corr(prediction& p1, prediction& p2, double base) {
    return preds_dot_prod(p1, p2, base) / sqrt(pred_var(p1, base) * pred_var(p2, base));
}

double pred_vecs_corr(
	std::vector<prediction>& ps1,
       	std::vector<prediction>& ps2, 
	double base,
       	bool calcCorrelation) {
    //calculate average correlation of the intersect.
    std::thread t1 {[&]{std::stable_sort(ps1.begin(), ps1.end(), pred_vid_lt<prediction>);}};
    std::thread t2 {[&]{std::stable_sort(ps2.begin(), ps2.end(), pred_vid_lt<prediction>);}};

    t1.join();
    t2.join();
 
    double c=0;
    int n=0;

    auto itr1 = ps1.begin();
    auto itr2 = ps2.begin();
    while (itr1 != ps1.end() && itr2 != ps2.end()) {
        int cmp = itr1->vid.compare(itr2->vid);
        if (cmp == 0) {
	    if (calcCorrelation)
	       c += preds_corr(*itr1, *itr2, base);
	    else
	       c += preds_dot_prod(*itr1, *itr2, base);
	    n += 1;
            itr1++;
            itr2++;
        }
        else if (cmp < 0)  //itr1->vid is lower
            itr1++;
        else  //itr2->vid is lower
            itr2++;
    }

    return c/n;
}

double pred_vecs_covar( std::vector<prediction>& ps1, std::vector<prediction>& ps2, double base) {
    return pred_vecs_corr(ps1, ps2, base, false);
}

double pred_vec_var( std::vector<prediction>& ps, double base) {
    double c = 0;
    for (auto it = ps.begin(); it != ps.end(); it++)
	c += pred_var(*it);
    return c/ps.size();
}

