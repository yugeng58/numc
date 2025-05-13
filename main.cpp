#include <iostream>
#include <stdexcept>
#include <string>
#include "omp.h"
#pragma omp requires unified_shared_memory

#include "ndarray_parallel.cpp"

using namespace std;

int main() {
    const int dim = 3;
    int shape_1[dim]{1,10,1};
    int shape_2[dim]{1,1,10};
    ndarray<int> A = ndarray<int>(dim,shape_1);
    ndarray<int> B = ndarray<int>(dim,shape_2);
    int* index_1 = new int[3]{};
    int* index_2 = new int[3]{};
    for(int i = 0;i < 10;i++){
        A.get_value(index_1) = i;
        B.get_value(index_2) = i;
        A.index_forward_iterate(index_1,dim);
        B.index_forward_iterate(index_2,dim);
    }
    delete [] index_1;
    delete [] index_2;

//    A.print();
//
//    B.print();

    A.matmul(B).print();

    A.matmul(B).axis_max(1).print();

    B.matmul(A).print();

//    A.axis_min(1).print();

    return 0;
}
