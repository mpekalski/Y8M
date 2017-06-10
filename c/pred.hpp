//
//  pred.hpp
//  yt8m_add_pred
//

#ifndef pred_hpp
#define pred_hpp

#include <stdio.h>
#include <string>
#include <map>
#include <vector>

// const
const int TOPK_INT = 120;

// utils
std::vector<std::pair<int,double> > map_2_ordered_pair(const std::map<int, double> & m);

// predictions, i/o, addition
struct prediction {
    std::string vid;
    std::map<int, double> probs;
};

template<typename T>
bool pred_vid_lt(const T& p1, const T& p2); //vid is less than, used for prediction and trueLabel

prediction add_2_preds(const prediction& x, const prediction& y, double w1=0.5, double w2=0.5, int topk=TOPK_INT);
prediction line2pred(const std::string & line);
std::vector <prediction> add_pred_vecs(std::vector<prediction>& p1, std::vector<prediction> & p2,
                                       double w1=0.5, double w2=0.5, int topk=TOPK_INT);


//geometric average
prediction gadd_2_preds(const prediction& x, const prediction& y, int topk=TOPK_INT, double floor=0.0001);
std::vector <prediction> gadd_pred_vecs(std::vector<prediction>& p1, std::vector<prediction> & p2,
                                        int topk=TOPK_INT, double floor=0.0001);

prediction line2pred(const std::string & line);

std::vector <prediction> file2preds(const std::string& filename);
std::string pred2line(const prediction& x, int topk=20);
int preds2file(const std::vector<prediction> & p, std::string& out_file, int topk=20);

// true labels, input
struct trueLabel {
    std::string vid;
    std::vector<int> labs;
};

trueLabel line2lab(const std::string& line);
std::vector<trueLabel> file2labels(const std::string& fn);

// Precision calculation
double precision(std::vector<int> p, std::vector<int> t);
double precision_one_example(prediction p, trueLabel t);
double GAP(std::vector<prediction> ps, std::vector<trueLabel> ts);

// Performance by label
std::map<int, double> label_performance_one_example(prediction p, const trueLabel& t);
std::map<int, std::pair<double, double> > LabelProbs(std::vector<prediction> ps, std::vector<trueLabel> ts);
#endif /* pred_hpp */
