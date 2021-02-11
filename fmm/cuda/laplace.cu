/*
CUDA kernels for Laplace problems.

(1) Gram Matrix kernel.

(2) P2P kernel.
*/

extern "C" {
    #define PI 3.141592654f
    #define TOL 1e-16

    __device__ double interact(double3 a, double3 b)
    {
        double3 r;
        r.x = a.x - b.x;
        r.y = a.y - b.y;
        r.z = a.z - b.z;

        double distSqr = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        if (r.x < TOL && r.y < TOL && r.z < TOL) {
            return 0.;
        } else {
            double invDistSqr = 1.0f/(4*PI*distSqr);
            return invDistSqr;
        }
     }

    __global__ void gram_matrix(double3* sources, double3* targets, double* result, int ntargets, int nsources)
    {

        int xid = blockDim.x * blockIdx.x + threadIdx.x;
        int yid = blockDim.y * blockIdx.y + threadIdx.y;

        if (xid < ntargets && yid < nsources){
            result[yid*nsources+xid] = interact(targets[yid], sources[xid]);
        }
    }
}

