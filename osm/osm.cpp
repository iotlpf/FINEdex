#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <climits>
#include <immintrin.h>
#include <cassert>
#include <random>
#include <memory>
#include <array>
#include <time.h>
#include <unistd.h>
#include <atomic>
#include <getopt.h>
#include <unistd.h>
#include <algorithm>
#include <mutex>

#include "function.h"
#include "finedex.h"


using namespace std;
using namespace finedex;

#define BUF_SIZE 2048


int main(int argc, char **argv) {
    parse_args(argc, argv);
    load_data();
    

    FINEdex<key_type, val_type> finedex;
    finedex.train(exist_keys, exist_keys, 32);


    // check find
    val_type dummy_val=0;
    int find_count=0;
    for(int i=0; i<exist_keys.size(); i++) {
        auto res = finedex.find(exist_keys[i], dummy_val);
        if(res==Result::ok) {
            find_count++;
            assert(dummy_val==exist_keys[i]);
        } else{
            LOG(5)<<"Non-find " <<i<<" : "<<exist_keys[i]<<endl;
            finedex.find_debug(exist_keys[i]);
            exit(0);
        }
    }
    LOG(3)<<"find ok: " << find_count;

    
    return 0;
}