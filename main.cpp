#include <iostream>
#include "omp.h"
#include "numc_safe.cpp"
using namespace std;

int main() {

    int shape[1] = {10};
    numc<int> A = numc<int>(1,shape);
    numc<int> B = numc<int>(1,shape);
    numc<int> C = numc<int>(1,shape);
    numc<int> D = numc<int>(1,shape);

    for(int i = 0;i < 10;i++){
        A[i].get() = i + 1;
        B[i].get() = i + 1;
        C[i].get() = i + 1;
        D[i].get() = i + 1;
    }


    A.reshape(4,new int[4]{10,1,1,1});
    B.reshape(4,new int[4]{1,10,1,1});
    C.reshape(4,new int[4]{1,1,10,1});
    D.reshape(4,new int[4]{1,1,1,10});
//    numc<int> E = A+B+C+D;

    numc<int> F = A.axis_max(0);

//    E = E.axis_max(0).axis_max(1).axis_max(2).axis_max(3);

//    E.print();

    return 0;
}
