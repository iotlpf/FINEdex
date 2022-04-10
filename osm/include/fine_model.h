#pragma once


#include "level_bin.h"
#include "lr_model.h"
#include "util.h"

namespace finedex{

template<class key_t, class val_t>
class FineModel {
public:
    typedef LinearRegressionModel<key_t> lrmodel_type;
    typedef LevelBin<key_t, val_t> levelbin_type;
    typedef FineModel<key_t, val_t> finemodel_type;

    typedef struct model_or_bin {
        typedef union pointer{
            levelbin_type* lb;
            finemodel_type* ai;
        }pointer_t;
        pointer_t mob;
        bool volatile isbin = true;   // true = lb, false = ai
        volatile uint8_t locked = 0;

        void lock(){
            uint8_t unlocked = 0, locked = 1;
            while (unlikely(cmpxchgb((uint8_t *)&this->locked, unlocked, locked) !=
                            unlocked))
              ;
        }
        void unlock(){
            locked = 0;
        }
    }model_or_bin_t;

private:
    lrmodel_type* model = nullptr;
    key_t* keys = nullptr;
    val_t* vals = nullptr;
    bool* valid_flag = nullptr;
    model_or_bin_t** mobs = nullptr;
    const size_t capacity;

    size_t maxErr=32;

public:
    explicit FineModel(double slope, double intercept, size_t epsilon,
                       const typename std::vector<key_t>::const_iterator &keys_begin,
                       const typename std::vector<val_t>::const_iterator &vals_begin, 
                       size_t size) : capacity(size), maxErr(epsilon)
    {
        model=new lrmodel_type(slope, intercept, epsilon);
        keys = (key_t *)malloc(sizeof(key_t)*size);
        vals = (val_t *)malloc(sizeof(val_t)*size);
        valid_flag = (bool*)malloc(sizeof(bool)*size);
        for(int i=0; i<size; i++){
            keys[i] = *(keys_begin+i);
            vals[i] = *(vals_begin+i);
            valid_flag[i] = true;
        }
        mobs = (model_or_bin_t**)malloc(sizeof(model_or_bin_t*)*(size+1));
        for(int i=0; i<size+1; i++){
            mobs[i]=nullptr;
        }
    }

    inline size_t get_capacity() {return capacity;}

    inline key_t get_lastkey() { return keys[capacity-1]; }

    inline key_t get_firstkey() { return keys[0]; }

    void print() {
        LOG(4)<<"[print finemodel] capacity:"<<capacity<<" -->";
        model->print();
        if(mobs[0]) {
            if(mobs[0]->isbin){
                mobs[0]->mob.lb->print(std::cout);
            }else {
                mobs[0]->mob.ai->print();
            }
        }
        for(size_t i=0; i<capacity; i++){
            std::cout<<"keys["<<i<<"]: " <<keys[i] << std::endl;
            if(mobs[i+1]) {
                if(mobs[i+1]->isbin){
                    mobs[i+1]->mob.lb->print(std::cout);
                }else {
                    mobs[i+1]->mob.ai->print();
                }
            }
        }
    }

    result_t find(const key_t &key, val_t &val)
    {
        size_t pos=0;
        if(find_array(key, pos)) {
            if(valid_flag[pos]){
                val=vals[pos];
                return result_t::ok;
            }
            return result_t::failed;
        }
        int bin_pos = pos;

        memory_fence();
        model_or_bin_t* mob = mobs[bin_pos];
        if(mob==nullptr) return result_t::failed;
    
        result_t res = result_t::failed;
        mob->lock();
        if(mob->isbin){
            res = mob->mob.lb->find(key, val);
        } else{
            res = mob->mob.ai->find(key, val);
        }
        assert(res!=result_t::retrain);
        mob->unlock();
        return res;
    }

    bool find_array(const key_t &key, size_t &pos) {
        auto [pre, lo, hi] = this->model->predict(key, capacity);
        assert(lo<capacity && hi-lo>=0);
        pos = binary_search_branchless(keys+lo, hi-lo, key) + lo;
        if(keys[pos]!=key) return false;
        return true; 
    }

    result_t find_debug(const key_t &key)
    {
        auto [pre, lo, hi] = this->model->predict(key, capacity);
        auto pos = binary_search_branchless(keys+lo, hi-lo, key);
        LOG(5) <<"key: "<<key <<", [pre, lo, hi, pos]: "<<pre<<", "<<lo<<", "<<hi<<", "<<pos;
        return result_t::ok;
    }

    void self_check()
    {
        for(size_t i=1; i<capacity; i++){
            assert(keys[i]>keys[i-1]);
            val_t dummy_val=0;
            auto res = find(keys[i], dummy_val);
            if(res!=Result::ok) {
                auto [pre, lo, hi] = this->model->predict(keys[i], capacity);
                LOG(5)<<"[fineModel error] i: "<< i<< ", key: "<<keys[i]<<" , [pre, lo, hi]: "<<pre<<", "<<lo<<", "<<hi;  
                exit(0);         
            }
            //assert(keys[i]==dummy_val);
        }
        for(size_t i=0; i<=capacity; i++){
            model_or_bin_t *mob = mobs[i];
            if(mob){
                if(mob->isbin){
                    mob->mob.lb->self_check();
                } else {
                    mob->mob.ai->self_check();
                }
            }
        }
    }

    // ======================= update =========================
    result_t update(const key_t &key, const val_t &val)
    {
        size_t pos=0;
        if(find_array(key, pos)) {
            if(valid_flag[pos]){
                vals[pos] = val;
                return result_t::ok;
            }
            return result_t::failed;
        }
        int bin_pos=pos;
        memory_fence();
        model_or_bin_t* mob = mobs[bin_pos];
        if(mob==nullptr) return result_t::failed;
    
        result_t res = result_t::failed;
        mob->lock();
        if(mob->isbin){
            res = mob->mob.lb->update(key, val);
        } else{
            res = mob->mob.ai->update(key, val);
        }
        assert(res!=result_t::retrain);
        mob->unlock();
        return res;
    }

    // =============================== insert =======================
    result_t insert(const key_t &key, const val_t &val)
    {
        size_t pos=0;
        if(find_array(key, pos)) {
            if(valid_flag[pos]){
                return result_t::failed;
            } else {
                valid_flag[pos] = true;
                vals[pos] = val;
                return result_t::ok;
            }
        }
        return insert_model_or_bin(key, val, pos);
    }

    // ========================== remove =====================
    result_t remove(const key_t &key)
    {
        size_t pos=0;
        if(find_array(key, pos)) {
            if(valid_flag[pos]){
                valid_flag[pos] = false;
                return result_t::ok;
            } 
            return result_t::failed;
        }
        return remove_model_or_bin(key, pos);
    }

    // ========================== scan ===================
    int scan(const key_t &key, const size_t n, std::vector<std::pair<key_t, val_t>> &result)
    {
        size_t remaining = n;
        size_t pos = 0;
        find_array(key, pos);
        while(remaining>0 && pos<=capacity) {
            if(pos<capacity && keys[pos]>=key){
                result.push_back(std::pair<key_t, val_t>(keys[pos], vals[pos]));
                remaining--;
                if(remaining<=0) break;
            }
            if(mobs[pos]!=nullptr){
                model_or_bin_t* mob = mobs[pos];
                if(mob->isbin){
                    remaining = mob->mob.lb->scan(key, remaining, result);
                } else {
                    remaining = mob->mob.ai->scan(key, remaining, result);
                }
            }
            pos++;
        }
        return remaining;
    }

private:
    inline size_t locate_in_levelbin(const key_t &key, const size_t pos)
    {
        // predict
        //size_t index_pos = model->predict(key);
        size_t index_pos = pos;
        size_t upbound = capacity-1;
        //index_pos = index_pos <= upbound? index_pos:upbound;

        // search
        size_t begin, end, mid;
        if(key > keys[index_pos]){
            begin = index_pos+1 < upbound? (index_pos+1):upbound;
            end = begin+maxErr < upbound? (begin+maxErr):upbound;
        } else {
            end = index_pos;
            begin = end>maxErr? (end-maxErr):0;
        }

        assert(begin<=end);
        while(begin != end){
            mid = (end + begin+2) / 2;
            if(keys[mid]<=key) {
                begin = mid;
            } else
                end = mid-1;
        }
        return begin;
    }

    result_t insert_model_or_bin(const key_t &key, const val_t &val, size_t bin_pos)
    {
        //LOG(5)<<"insert key: "<<key<< ", into bin: "<<bin_pos;
        // insert bin or model
        model_or_bin_t *mob = mobs[bin_pos];
        if(mob==nullptr){
            mob = new model_or_bin_t();
            mob->lock();
            mob->mob.lb = new levelbin_type();
            mob->isbin = true;
            memory_fence();
            if(mobs[bin_pos]){
                delete mob;
                return insert_model_or_bin(key, val, bin_pos);
            }
            mobs[bin_pos] = mob;
        } else{
            mob->lock();
        }
        assert(mob!=nullptr);
        assert(mob->locked == 1);
        result_t res = result_t::failed;
        if(mob->isbin) {           // insert into bin
            res = mob->mob.lb->insert(key, val);
            if(res == result_t::retrain){
                //LOG(5)<<"[finemodel] Need Retrain: " << key;
                // resort the data and train the model
                std::vector<key_t> retrain_keys;
                std::vector<val_t> retrain_vals;
                mob->mob.lb->resort(retrain_keys, retrain_vals);
                lrmodel_type model(0.0, 0.0, 0);
                model.train(retrain_keys.begin(), retrain_keys.size());
                finemodel_type *ai = new finemodel_type(model.get_slope(), model.get_intercept(), model.get_epsilon(),
                                                        retrain_keys.begin(), retrain_vals.begin(), retrain_keys.size());
                
                memory_fence();
                delete mob->mob.lb;
                mob->mob.ai = ai;
                mob->isbin = false;
                res = ai->insert(key, val);
                mob->unlock();
                return res;
            }
        } else{                   // insert into model
            res = mob->mob.ai->insert(key, val);
        }
        mob->unlock();
        //print();
        return res;
    }

    result_t remove_model_or_bin(const key_t &key, const int bin_pos)
    {
        memory_fence();
        model_or_bin_t* mob = mobs[bin_pos];
        if(mob==nullptr) return result_t::failed;

        result_t res = result_t::failed;
        mob->lock();
        if(mob->isbin){
            res = mob->mob.lb->remove(key);
        } else{
            res = mob->mob.ai->remove(key);
        }
        assert(res!=result_t::retrain);
        mob->unlock();
        return res;
    }

};

} //namespace findex


