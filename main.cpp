#include <iostream>
#include "omp.h"
#include "numc_modify.cpp"
using namespace std;

int main() {

    int shape[1] = {10};
    numc<int> A = numc<int>(1,shape);
    numc<int> B = numc<int>(1,shape);
    numc<int> C = numc<int>(1,shape);
    numc<int> F = numc<int>(1,shape);

    for(int i = 0;i < 10;i++){
        A[i].get() = i + 1;
        B[i].get() = i + 1;
        C[i].get() = i + 1;
    }


    A.reshape(3,new int[3]{10,1,1});
    B.reshape(3,new int[3]{1,10,1});
    C.reshape(3,new int[3]{1,1,10});
    numc<int> D = A*B*C;

    D.print();

    D.slice(0,5,1).print();

    return 0;

}
