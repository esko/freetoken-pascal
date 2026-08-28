#include <cuda_runtime.h>

__global__ void freetoken_sm61_probe(int *output) {
    if (threadIdx.x == 0) {
        *output = 61;
    }
}

int main() {
    return 0;
}
