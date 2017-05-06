#define argument 375

void GetSemaphor(__global int * semaphor) {
   int occupied = atom_xchg(semaphor, 1);
   while(occupied > 0)
   {
     occupied = atom_xchg(semaphor, 1);
   }
}

void ReleaseSemaphor(__global int * semaphor)
{
   int prevVal = atom_xchg(semaphor, 0);
}

int rand_r (int *seed){
      int next = *seed;
      int result;

      next *= 1103515245;
      next += 12345;
      result = (unsigned int) (next / 65536) % 2048;

      next *= 1103515245;
      next += 12345;
      result <<= 10;
      result ^= (unsigned int) (next / 65536) % 1024;

      next *= 1103515245;
      next += 12345;
      result <<= 10;
      result ^= (unsigned int) (next / 65536) % 1024;

      return result;
}

void shiftDown(__global double *arr, int root, int bottom){
    int largest = root;  // Initialize largest as root
    int l = 2*root + 1;  // left = 2*i + 1
    int r = 2*root + 2;  // right = 2*i + 2

    if (l < bottom && arr[l] > arr[largest])
        largest = l;

    if (r < bottom && arr[r] > arr[largest])
        largest = r;

    if (largest != root)
    {
        double temp = arr[root];
        arr[root] = arr[largest];
        arr[largest] = temp;
        shiftDown(arr, largest, bottom);
    }
}

__kernel void gen1(__global double* M1, __global double* M2, size_t N, size_t N2){
	int idx = get_global_id(0);
	if (idx < N){
		int arg = argument;	
		M1[idx] = (rand_r(&idx)%100) *0.01;
	}
	if (idx < N2){
                int arg = argument;
                int max_arg = argument*9;
                int rand = rand_r(&idx);
                M2[idx] = (rand%max_arg)+arg+ (rand == max_arg-1 ? 1 :0);
        }

}

__kernel void gen2(__global double* M2, size_t N){
	int idx = get_global_id(0);
	if (idx < N){
		int arg = argument;
		int max_arg = argument*9;
		int rand = rand_r(&idx);
		M2[idx] = (rand%max_arg)+arg+ (rand == max_arg-1 ? 1 :0);
	}
}

__kernel void map1(__global double* M1, size_t N){
	int idx = get_global_id(0);
	if (idx < N){
		M1[idx] = atanh(M1[idx]);
	}
}

__kernel void map2(__global double* M2, size_t N){
	for (int i = 1;i<N;++i){
		M2[i] += M2[i-1];
	}
}

__kernel void map3(__global double* M2, size_t N){
	int idx = get_global_id(0);
	if (idx < N){
		M2[idx] = fabs(atan(M2[idx]));
	}
}

__kernel void merge(__global double* M1, __global double* M2, size_t N){
	int idx = get_global_id(0);
	if (idx < N){
		M2[idx] *= M1[idx];
	}
}

__kernel void heap(__global double* M2, size_t N){
	int idx = get_global_id(0);
	for (int i = (N/2)-1-idx; i>=3;i -= 4){
		shiftDown(M2,i,(int)N);
	}
}

__kernel void sort(__global double* M2, size_t N){
        for (int i = N - 1; i >= 1; --i){
		double temp = M2[0];
                M2[0] = M2[i];
                M2[i] = temp;
                shiftDown(M2, 0, i);
        }
}

__kernel void min_elem(__global double* M2, size_t N, __global double* min){
	for (int i = 3; i>=0; --i)
              	shiftDown(M2,i,(int)N);
	for (int i=0;i<N;++i){
		if (M2[i] != 0){
			*min = M2[i];
			break;
		}
	}
}

__kernel void reduce(__global double* M2, size_t N, __global double* min, __global double* result,__global int *semaphor){
	int idx = get_global_id(0);
	if (idx < N){
		if ((int)(M2[idx]/(*min))%2 == 0){
			while (M2[idx] > 1.00){
				M2[idx] -= 1;
			}
 			GetSemaphor(semaphor);
                        *result += asin(M2[idx]);
			ReleaseSemaphor(semaphor);
		}
	}
}

