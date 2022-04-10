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

  // used for top level
  struct Segment;
  size_t n;
  key_t first_key;
  size_t EpsilonRecursive=4;
  std::vector<key_t> model_keys_for_model;
  std::vector<Segment> segments;
  std::vector<size_t> levels_offsets;

  // used for bottom level
  size_t Epsilon;
  std::vector<key_t> model_keys;
  std::vector<finemodel_type> models;
  

 public:
  inline FINEdex() : model_keys(), model_keys_for_model(), models(), segments(), levels_offsets(), 
                     first_key(key_t(0)), EpsilonRecursive(4) {}

  void train(const std::vector<key_t> &keys, 
             const std::vector<val_t> &vals, size_t epsilon)
  {
    assert(keys.size() == vals.size());
    if(keys.size()==0) return;
    this->Epsilon = epsilon;
    LOG(2) << "Training data: "<<keys.size()<<" ,Epsilon: "<<Epsilon;

    OptimalPLR* opt = new OptimalPLR(Epsilon-1);
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
        //models.emplace_back(cs_slope, cs_intercept, Epsilon, k_iter, v_iter, pos);
        append_model(cs_slope, cs_intercept, Epsilon, k_iter, v_iter, pos);
        k_iter += pos;
        v_iter += pos;
        pos=0;
        opt = new OptimalPLR(Epsilon-1);
        opt->add_point(p, pos);
      }
    }
    auto cs = opt->get_segment();
    auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
    //models.emplace_back(cs_slope, cs_intercept, Epsilon, k_iter, v_iter, ++pos);
    append_model(cs_slope, cs_intercept, Epsilon, k_iter, v_iter, ++pos);

    LOG(2) << "Training models: "<<models.size();
    self_check();
    LOG(2)<<"Self check: OK";
    assert(model_keys.size() == models.size());
    build(model_keys_for_model.begin(), model_keys_for_model.end(), EpsilonRecursive);
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
      auto[pos, lo, hi] = search(key);
      pos = binary_search_branchless(&model_keys_for_model[lo], hi-lo, key)+lo;
      while(remaining>0 && pos < models.size()){
        remaining = models[pos].scan(key, remaining, result);
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

  
 private:
  void append_model(double slope, double intercept, size_t epsilon,
                     const typename std::vector<key_t>::const_iterator &keys_begin,
                     const typename std::vector<val_t>::const_iterator &vals_begin, 
                     size_t size) 
  {
    models.emplace_back(slope, intercept, epsilon, keys_begin, vals_begin, size);
    model_keys.push_back(models.back().get_lastkey());
    model_keys_for_model.push_back(models.back().get_lastkey());
  }

  finemodel_type* find_model(const key_t &key)
  {
    size_t model_pos = binary_search_branchless(&model_keys[0], model_keys.size(), key);
    if(model_pos >= models.size())
      model_pos = models.size()-1;
    return &models[model_pos];
    
    // root for model
    auto[pos, lo, hi] = search(key);
    pos = binary_search_branchless(&model_keys_for_model[lo], hi-lo, key)+lo;
    pos = pos < models.size()? pos : models.size()-1;
    return &models[pos];
  }

  template<typename RandomIt>
  void build(RandomIt first, RandomIt last, size_t epsilon_recursive)
  {
    n = (size_t) std::distance(first, last);
    if (n == 0)
        return;
    this->first_key = *first;

    levels_offsets.push_back(0);
    segments.reserve(n / (epsilon_recursive * epsilon_recursive));

    auto ignore_last = *std::prev(last) == std::numeric_limits<key_t>::max(); // max() is the sentinel value
    auto last_n = n - ignore_last;
    last -= ignore_last;

    auto build_level = [&](auto epsilon, auto in_fun, auto out_fun) {
        auto n_segments = make_segmentation(last_n, epsilon, in_fun, out_fun);
        if (segments.back().slope == 0 && last_n > 1) {
            // Here we need to ensure that keys > *(last-1) are approximated to a position == prev_level_size
            segments.emplace_back(*std::prev(last) + 1, 0, last_n);
            ++n_segments;
        }
        segments.emplace_back(last_n); // Add the sentinel segment
        return n_segments;
    };

    // Build first level
    auto in_fun = [&](auto i) {
        auto x = first[i];
        // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
        // to x=first[i] such that x+1!=first[i+1], we map the values x+1,...,first[i+1]-1 to their correct rank i
        auto flag = i > 0 && i + 1u < n && x == first[i - 1] && x != first[i + 1] && x + 1 != first[i + 1];
        return std::pair<key_t, size_t>(x + flag, i);
    };
    auto out_fun = [&](auto cs) { segments.emplace_back(cs); };
    last_n = build_level(epsilon_recursive, in_fun, out_fun);
    levels_offsets.push_back(levels_offsets.back() + last_n + 1);

    // Build upper levels
    while (epsilon_recursive && last_n > 1) {
      auto offset = levels_offsets[levels_offsets.size() - 2];
      auto in_fun_rec = [&](auto i) { return std::pair<key_t, size_t>(segments[offset + i].key, i); };
      last_n = build_level(epsilon_recursive, in_fun_rec, out_fun);
      levels_offsets.push_back(levels_offsets.back() + last_n + 1);
    }
  }

  auto segment_for_key(const key_t &key) const {
    if (EpsilonRecursive == 0) {
        return std::prev(std::upper_bound(segments.begin(), segments.begin() + segments_count(), key));
    }

    auto it = segments.begin() + *(levels_offsets.end() - 2);
    for (auto l = int(height()) - 2; l >= 0; --l) {
        auto level_begin = segments.begin() + levels_offsets[l];
        auto pos = std::min<size_t>((*it)(key), std::next(it)->intercept);
        auto lo = level_begin + SUB_EPS(pos, EpsilonRecursive + 1);

        static constexpr size_t linear_search_threshold = 8 * 64 / sizeof(Segment);
        if (EpsilonRecursive <= linear_search_threshold) {
            for (; std::next(lo)->key <= key; ++lo)
                continue;
            it = lo;
        } else {
            auto level_size = levels_offsets[l + 1] - levels_offsets[l] - 1;
            auto hi = level_begin + ADD_EPS(pos, EpsilonRecursive, level_size);
            it = std::prev(std::upper_bound(lo, hi, key));
        }
    }
    return it;
  }

  size_t segments_count() const { return segments.empty() ? 0 : levels_offsets[1] - 1; }

  size_t height() const { return levels_offsets.size() - 1; }

  ApproxPos search(const key_t &key) const {
      auto k = std::max(first_key, key);
      auto it = segment_for_key(k);
      auto pos = std::min<size_t>((*it)(k), std::next(it)->intercept);
      auto lo = SUB_EPS(pos, Epsilon);
      auto hi = ADD_EPS(pos, Epsilon, n);
      return {pos, lo, hi};
  }

};


#pragma pack(push, 1)

template<class key_t, class val_t>
struct FINEdex<key_t, val_t>::Segment {
    key_t key;             ///< The first key that the segment indexes.
    float slope;    ///< The slope of the segment.
    int32_t intercept; ///< The intercept of the segment.

    Segment() = default;

    Segment(key_t key, float slope, int32_t intercept) : key(key), slope(slope), intercept(intercept) {};

    explicit Segment(size_t n) : key(std::numeric_limits<key_t>::max()), slope(), intercept(n) {};

    explicit Segment(const typename PLR<key_t, size_t>::CanonicalSegment &cs)
        : key(cs.get_first_x()) {
      auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(key);
      if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
          throw std::overflow_error("Change the type of Segment::intercept to int64");
      slope = cs_slope;
      intercept = cs_intercept;
    }

    friend inline bool operator<(const Segment &s, const key_t &k) { return s.key < k; }
    friend inline bool operator<(const key_t &k, const Segment &s) { return k < s.key; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.key < t.key; }

    operator key_t() { return key; };

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t operator()(const key_t &k) const {
      auto pos = int64_t(slope * (k - key)) + intercept;
      return pos > 0 ? size_t(pos) : 0ull;
    }
};

#pragma pack(pop)


}// namespace finedex