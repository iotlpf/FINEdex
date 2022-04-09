#pragma once

#include "util.h"
#include "fine_model.h"
#include "plr.hpp"

namespace finedex{

template<class key_t, class val_t>
class FINEdex{
 private:
  typedef FineModel<key_t, val_t> finemodel_type;
  typedef PLR<key_t, size_t> OptimalPLR;
  typedef typename PLR<key_t, size_t>::CanonicalSegment canonical_segment;
    
  size_t Epsilon;
  std::vector<key_t> model_keys;
  std::vector<finemodel_type> models;
  std::vector<canonical_segment> segments;

 public:
  inline FINEdex() : model_keys(), segments() {}

  void train(const std::vector<key_t> &keys, 
             const std::vector<val_t> &vals, size_t epsilon)
  {
    assert(keys.size() == vals.size());
    if(keys.size()==0) return;
    this->Epsilon = epsilon;
    LOG(2) << "Training data: "<<keys.size()<<" ,Epsilon: "<<Epsilon;

    OptimalPLR* opt = new OptimalPLR(Epsilon);
    key_t p = keys[0];
    size_t pos=0;
    opt->add_point(p, pos);
    auto k_iter = keys.begin();
    auto v_iter = vals.begin();
    for(int i=1; i<keys.size(); i++) {
      key_t next_p = keys[i];
      if (next_p == p){
        LOG(5)<<"DUPLICATE keys";
        exit(0);
      }
      p = next_p;
      pos++;
      if(!opt->add_point(p, pos)) {
        auto cs = opt->get_segment();
        auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
        models.emplace_back(cs_slope, cs_intercept, Epsilon, k_iter, v_iter, pos);
        k_iter += pos;
        v_iter += pos;
        pos=0;
        opt = new OptimalPLR(Epsilon);
        opt->add_point(p, pos);
      }
    }
    auto cs = opt->get_segment();
    auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
    models.emplace_back(cs_slope, cs_intercept, Epsilon, k_iter, v_iter, ++pos);

    LOG(2) << "Training models: "<<models.size();
    self_check();
    LOG(2)<<"Self check: OK";
    for(int i=0; i<models.size(); i++) model_keys.push_back(models[i].get_lastkey());
    assert(model_keys.size() == models.size());
    //print();
  }

  void self_check() {
    for(int i=0; i<model_keys.size(); i++){
      models[i].self_check();
    }
  }

  void print() {
    for(int i=0; i<model_keys.size(); i++){
      LOG(3)<<"model "<<i<<" ,key:"<<model_keys[i]<<" ->";
      models[i].print();
    }
  }

  // =================== search the data =======================
  inline result_t find(const key_t &key, val_t &val)
  {  
    return find_model(key)[0].find(key, val);
  }

  inline result_t find_debug(const key_t &key) {
    return find_model(key)[0].find_debug(key);
  }

  // =================  scan ====================
  int scan(const key_t &key, const size_t n, std::vector<std::pair<key_t, val_t>> &result)
  {
      size_t remaining = n;
      size_t model_pos = binary_search_branchless(&model_keys[0], model_keys.size(), key);
      if(model_pos >= models.size())
          model_pos = models.size()-1;
      while(remaining>0 && model_pos < models.size()){
          remaining = models[model_pos].scan(key, remaining, result);
      }
      return remaining;
  }

  // =================== insert the data =======================
  inline result_t insert(const key_t& key, const val_t& val)
  {
      return find_model(key)[0].insert(key, val);
  }

  // ================ update =================
  inline result_t update(const key_t& key, const val_t& val)
  {
      return find_model(key)[0].update(key, val);
  }


  // ==================== remove =====================
  inline result_t remove(const key_t& key)
  {
      return find_model(key)[0].remove(key);
  }

  /*inline result_t find(const key_t &key, val_t &val);
    inline result_t insert(const key_t &key, const val_t &val);
    inline result_t update(const key_t &key, const val_t &val);
    inline result_t remove(const key_t &key);
    int scan(const key_t &key, const size_t n, std::vector<std::pair<key_t, val_t>> &result);
    size_t model_size();*/

  
 private:
  finemodel_type* find_model(const key_t &key)
  {
      // root 
      size_t model_pos = binary_search_branchless(&model_keys[0], model_keys.size(), key);
      if(model_pos >= models.size())
          model_pos = models.size()-1;
      return &models[model_pos];
  }

};


}// namespace finedex