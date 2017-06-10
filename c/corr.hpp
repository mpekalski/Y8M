//
//  pred.hpp
//  yt8m_add_pred
//

#ifndef corr_hpp
#define corr_hpp

#include <stdio.h>
#include <string>
#include <map>
#include <vector>

#include "pred.hpp"

const double BASE=2.12e-4; // 1/4716 prob if random labeling.
double pred_var(const prediction& p, double base=BASE);
double preds_dot_prod(prediction& p1, prediction& p2, double base=BASE); 
double preds_corr(prediction& p1, prediction& p2, double base=BASE);

double pred_vecs_corr(std::vector<prediction> & ps1, std::vector<prediction> & ps2,
       double base=BASE, bool calcCorrelation=true); //if false, calculate covariance
double pred_vecs_covar( std::vector<prediction>& ps1, std::vector<prediction>& ps2, double base=BASE);
double pred_vec_var( std::vector<prediction>& ps, double base=BASE);

#endif
