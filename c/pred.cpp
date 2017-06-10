//
//  pred.cpp
//  yt8m_add_pred
//


#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <cmath>

#include "pred.hpp"
#include "utils.hpp"

//*** utils  ***//

std::vector<std::pair<int,double> > map_2_ordered_pair(const std::map<int, double> & m) {
    // order the probs from large to small.
    std::vector<std::pair<int, double>> pairs;
    for (auto itr = m.begin(); itr != m.end(); ++itr)
        pairs.push_back(*itr);
    std::sort(pairs.begin(), pairs.end(),
              [=](std::pair<int, double>& a, std::pair<int, double>& b)
              {return a.second > b.second;} );
    return pairs;
}

//*** predictions  ***//
prediction line2pred(const std::string & line) {
    
    prediction x;
    std::stringstream ss(line);
    
    getline(ss, x.vid, ',');
    
    std::string tmp1, tmp2;
    while(ss) {
        getline(ss, tmp1, ' ');
        getline(ss, tmp2, ' ');
        x.probs[stoi(tmp1)] = stod(tmp2);
    }
    
    return x;
}

std::string pred2line(const prediction& x, int topk) {
    std::string s = x.vid + ',';
    
    std::vector<std::pair<int, double>> pairs = map_2_ordered_pair(x.probs);
    
    auto it = pairs.begin();
    for (int i=0; i<pairs.size()-1 && i<topk-1; i++, it++)
        s += std::to_string(it->first) + ' ' + std::to_string(it->second)+' ';
    s += std::to_string(it->first) + ' ' + std::to_string(it->second)+'\n';
     
    return s;    
}

template<typename T>
bool pred_vid_lt(const T& p1, const T& p2) {
    return p1.vid.compare(p2.vid) < 0;
}

std::vector <prediction> file2preds(const std::string& filename) {
    //std::cout<<"processing "<<filename<<'\n';
    std::ifstream f;
    f.open(filename);
    
    std::string tmp;
    getline(f, tmp); //discard the first line
    
    // separate reading and converting for concurrency
    std::vector <std::string> lines;
    while(getline(f, tmp))
        lines.push_back(tmp);
    
    std::vector<prediction> x;
    for(auto line : lines)
	x.push_back(line2pred(line));
    
    return x;
}

prediction add_2_preds(const prediction & x, const prediction & y, double w1, double w2, int topk) {
    if (x.vid.compare(y.vid) != 0) {
        std::cout<<"incompatible video ids: "<<x.vid<<", "<<y.vid<<'\n';
        exit(1);
    }
    
    std::map<int, double> tmp_probs;
    for (auto it = x.probs.begin(); it != x.probs.end(); ++it)
        tmp_probs[it->first] = w1 * it->second;
    
    for (auto it = y.probs.begin(); it != y.probs.end(); ++it)
        tmp_probs[it->first] += w2 * it->second;
    
    // order the probs from large to small.
    std::vector<std::pair<int, double>> pairs = map_2_ordered_pair(tmp_probs);
    
    prediction z;
    z.vid = x.vid;
    for(int i=0; i<std::min<int>(topk, static_cast<int>(pairs.size())); i++)
        z.probs[pairs[i].first] = pairs[i].second;

    return z;
}

std::vector <prediction> add_pred_vecs(std::vector<prediction>& p1, std::vector<prediction> & p2,
                                       double w1, double w2, int topk)
{
    std::thread t1 {[&]{std::stable_sort(p1.begin(), p1.end(), pred_vid_lt<prediction>);}};
    std::thread t2 {[&]{std::stable_sort(p2.begin(), p2.end(), pred_vid_lt<prediction>);}};

    t1.join();
    t2.join();
    std::cout<<currentDateTime()<<": Pred vecs sorted...\n";
    
    std::vector<prediction> p;
    
    auto itr1 = p1.begin();
    auto itr2 = p2.begin();
    while (itr1 != p1.end() && itr2 != p2.end()) {
        int cmp = itr1->vid.compare(itr2->vid);
        if (cmp == 0) {
            p.push_back(add_2_preds(*itr1, *itr2, w1, w2, topk));
            itr1++;
            itr2++;
        }
        else if (cmp < 0) { //itr1->vid is lower, keep wgt = w1 + w2
            p.push_back(add_2_preds(*itr1, *itr1, w1, w2, topk));
            itr1++;
        }
        else { //itr2->vid is lower
            p.push_back(add_2_preds(*itr2, *itr2, w1, w2, topk));
            itr2++;
        }
    }
    for (;itr1 != p1.end(); itr1++)
        p.push_back(add_2_preds(*itr1, *itr1, w1, w2, topk));
    for (;itr2 != p2.end(); itr2++)
        p.push_back(add_2_preds(*itr2, *itr2, w1, w2, topk));
    
    return p;
}


prediction gadd_2_preds(const prediction & x, const prediction & y,
       int topk, double floor) {
    if (x.vid.compare(y.vid) != 0) {
        std::cout<<"incompatible video ids: "<<x.vid<<", "<<y.vid<<'\n';
        exit(1);
    }
    
    std::map<int, double> tmp_probs = x.probs;
    
    for (auto it = y.probs.begin(); it != y.probs.end(); ++it)
        tmp_probs[it->first] = sqrt(std::min<double>(floor, tmp_probs[it->first]) * it->second);
    
    // order the probs from large to small.
    std::vector<std::pair<int, double>> pairs = map_2_ordered_pair(tmp_probs);
    
    prediction z;
    z.vid = x.vid;
    for(int i=0; i<std::min<int>(topk, static_cast<int>(pairs.size())); i++)
        z.probs[pairs[i].first] = pairs[i].second;

    return z;
}

std::vector <prediction> gadd_pred_vecs(std::vector<prediction>& p1, std::vector<prediction> & p2,
                                        int topk, double floor)
{
    std::thread t1 {[&]{std::stable_sort(p1.begin(), p1.end(), pred_vid_lt<prediction>);}};
    std::thread t2 {[&]{std::stable_sort(p2.begin(), p2.end(), pred_vid_lt<prediction>);}};

    t1.join();
    t2.join();
    std::cout<<currentDateTime()<<": Pred vecs sorted...\n";
    
    std::vector<prediction> p;
    
    auto itr1 = p1.begin();
    auto itr2 = p2.begin();
    while (itr1 != p1.end() && itr2 != p2.end()) {
        int cmp = itr1->vid.compare(itr2->vid);
        if (cmp == 0) {
            p.push_back(gadd_2_preds(*itr1, *itr2, topk,floor));
            itr1++;
            itr2++;
        }
        else if (cmp < 0) { //itr1->vid is lower
            p.push_back(*itr1);
            itr1++;
        }
        else { //itr2->vid is lower
            p.push_back(*itr2);
            itr2++;
        }
    }
    for (;itr1 != p1.end(); itr1++)
        p.push_back(*itr1);
    for (;itr2 != p2.end(); itr2++)
        p.push_back(*itr2);
    
    return p;
}

int preds2file(const std::vector<prediction> & p, std::string& out_file, int topk) {
    std::ofstream f;
    f.open(out_file);
    
    f<<"VideoId,LabelConfidencePairs\n";
    for (auto x : p)
        f << pred2line(x, topk);

    return static_cast<int>(p.size());
}

//*** true labels  ***//

trueLabel line2lab(const std::string & line) {
    trueLabel x;
    std::stringstream ss(line);
    
    getline(ss, x.vid, ',');
    
    int tmp;
    while(ss >> tmp)
        x.labs.push_back(tmp);
    
    return x;
}

std::vector<trueLabel> file2labels(const std::string& fn) {
    std::ifstream f;
    f.open(fn);
    std::vector<trueLabel> ts;
    std::vector<std::string> lines;

    std::string tmp;
    while( getline(f, tmp))
	lines.push_back(tmp);

    for(auto line : lines)
        ts.push_back(line2lab(line));
    
    return ts;
}

//*** GAP calculation ***//
double precision(std::vector<int> p, std::vector<int> t) {
    //precision of predicted labels ref. true labels.
    //p is sorted by descending probability.

    double delta_recall= 1. / t.size();
    double precn = 0.0;
    double poscount = 0.0;
    for (int i=0; i<p.size(); i++) {
        if (std::find(t.begin(), t.end(), p[i]) != t.end()) {
            poscount += 1.0;
            precn += poscount / (i+1.0);
        }
    }
    return delta_recall * precn;
}

double precision_one_example(prediction p, trueLabel t) {
    if (p.vid.compare(t.vid) != 0) {
        std::cout<<"Incompatible vids between prediction and trueLabel: "<<p.vid<<", "<<t.vid<<'\n';
        exit(1);
    }
    std::vector< std::pair<int, double> > sortPred = map_2_ordered_pair(p.probs);
    std::vector<int> plabs;
    for(auto it=sortPred.begin(); it != sortPred.end(); it++)
        plabs.push_back(it->first);
    
    return precision(plabs, t.labs);
}

double GAP(std::vector<prediction> ps, std::vector<trueLabel> ts) {
    std::stable_sort(ps.begin(), ps.end(), pred_vid_lt<prediction>);
    std::stable_sort(ts.begin(), ts.end(), pred_vid_lt<trueLabel>);
    
    auto itp = ps.begin();
    auto itt = ts.begin();
    double gap = 0.0;
    int count = 0;

    while (itp != ps.end() && itt != ts.end()) {
        int cmp = itp->vid.compare(itt->vid);
        if (cmp == 0) {
	   gap += precision_one_example(*itp, *itt); 
	   count += 1;
           itp++;
           itt++;
        }
        else if (cmp < 0)  //itp->vid is lower
            itp++;
        else  //itr2->vid is lower
            itt++;
    }
    
    return gap/count;
}

///***   performance by true labels  ***///
std::map<int, double> label_performance_one_example(prediction p, const trueLabel& t) {
    if (p.vid.compare(t.vid) != 0) {
        std::cout<<"Incompatible vids between prediction and trueLabel: "<<p.vid<<", "<<t.vid<<'\n';
        exit(1);
    }
    std::map<int, double> true_lab_prob;
    for(auto it=t.labs.begin(); it != t.labs.end(); it++)
	true_lab_prob[*it] = p.probs[*it];
    
    return true_lab_prob;
}

std::map<int, std::pair<double, double> > LabelProbs(std::vector<prediction> ps, std::vector<trueLabel> ts)
{
    std::stable_sort(ps.begin(), ps.end(), pred_vid_lt<prediction>);
    std::stable_sort(ts.begin(), ts.end(), pred_vid_lt<trueLabel>);
    
    auto itp = ps.begin();
    auto itt = ts.begin();
   
    std::map<int, std::pair<double, double> > label_probs; // label, count, cumu_prob    
    for ( ; itp != ps.end() && itt != ts.end(); itp++, itt++) {
       std::map<int, double> ex_probs = label_performance_one_example(*itp, *itt); 
       for (auto e : ex_probs) {
	   label_probs[e.first].first += 1.0;
	   label_probs[e.first].second += e.second;
       }
    }
    return label_probs;
} 
