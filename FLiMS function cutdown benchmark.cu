/*
Copyright 2024 Colm Agnew Gallagher

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <iostream>

void print_array(const char* name, int* array, size_t size) 
{
    printf("%s: ", name);
    for (size_t i = 0; i < size; ++i) 
	{
        printf("%d ", array[i]);
    }
    printf("\n");
}

void generateArray(int* array, int size) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(INT_MIN , INT_MAX);

    //fill array with random numbers
    for (int i = 0; i < size; ++i) 
    {
        array[i] = dist(gen);
    }

	//sort array in descending order
    std::sort(array, array + size, std::greater<int>());
}


bool isSorted(int* arr, int size) 
{
    for (int i = 0; i < size - 1; ++i) 
	{
        if (arr[i] < arr[i + 1]) 
		{ 
            printf("Order violation: arr[%d] = %d, arr[%d] = %d\n", i, arr[i], i+1, arr[i+1]);
            return false; 
        }
    }
    return true;
}

int countZeros(int* arr, int size) 
{
    int count = 0;
    for (int i = 0; i < size; ++i) 
	{
        if (arr[i] == 0) 
		{
            count++;
        }
    }
    return count;
}

int countDupes(int* arr, int size) 
{
    int count = 0;
    for (int i = 0; i < size - 1; ++i) 
{
        if (arr[i] == arr[i + 1]) 
		{
			//printf("Dupe found: %d\n", arr[i]);
            count++;
        }
    }
    return count;
}


__global__ void max_unit_kernel_AB_4(int *A, int *B, int *C, int *UsedA, int *UsedB, int *a_indices, int *b_indices, int size) 
{
	int i = threadIdx.x;
    //inside kernel loop
    for(int offset = 0; offset < size * 2; offset += 4) 
	{
		//check to see if inbounds
        if (i + offset < size * 2) 
		{
			//set state from bitmap
            int a_index = a_indices[i];
            int b_index = b_indices[i];

            //set bitmap value of index to 1 to show it is in usage
            if (a_index < size) atomicCAS(&UsedA[a_index], 0, 1);
            if (b_index < size) atomicCAS(&UsedB[b_index], 0, 1);
			
			//set state from bitmap
            int a_state = UsedA[a_index]; 
            int b_state = UsedB[b_index];

            if (a_state == 1 && (b_state != 1 || A[a_index] >= B[b_index])) 
			{
				//set c array to value
                C[i + offset] = A[a_index];
				//set used to 2 showing that its been used
                UsedA[a_index] += 1;
				//check to see if new index value is less than size , if so switch to it otherwise don't change
                a_indices[i] = (a_index + 4 < size) ? a_index + 4 : a_index;
            } 
			else if (b_state == 1) 
			{
				//set c array to value
                C[i + offset] = B[b_index];
				//set used to 2 showing that its been used
                UsedB[b_index] += 1;
				//check to see if new index value is less than size , if so switch to it otherwise don't change
                b_indices[i] = (b_index + 4 < size) ? b_index + 4 : b_index;
            }
			printf("i: %d, A[%d]: %d, B[%d]: %d\n", i, a_index, A[a_index], b_index, B[b_index]);
        }
    }
}



__global__ void cas_kernel_4(int *C, int size) 
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int offset = blockId * blockDim.x;
    
    while (offset < size) 
    {
        __shared__ int temp[4];
        int idx = threadIdx.x;

        //offset is within bounds 
        if (offset + idx < size) 
        {
            temp[idx] = C[offset + idx];
        }

        __syncthreads();

        // CAS with 2
        int ixj = idx ^ 2;
        if ((idx < ixj) && ((idx & 2) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
        {
			printf("Block:%d Thread:%d CAS2 swap temp[%d] = %d, temp[%d] = %d are going to be swapped\n", blockId, idx, idx, temp[idx], ixj, temp[ixj]);
            int tmp = temp[idx];
            temp[idx] = temp[ixj];
            temp[ixj] = tmp;
        }

        __syncthreads();

        // CAS with 1
        ixj = idx ^ 1;
        if ((idx < ixj) && ((idx & 1) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
        {
			printf("Block:%d Thread:%d CAS1 swap temp[%d] = %d, temp[%d] = %d are going to be swapped\n", blockId, idx, idx, temp[idx], ixj, temp[ixj]);
            int tmp = temp[idx];
            temp[idx] = temp[ixj];
            temp[ixj] = tmp;
        }

        if (offset + idx < size) 
        {
            C[offset + idx] = temp[idx];
        }

        offset += blockDim.x * gridDim.x; 
        __syncthreads(); 
    }
}






int main() 
{
	//65536 16384 32768 16777216 1048576 131072 262144 524288 1048576 2097152 4194304 8388608 , 33554432 , 67108864 , 134217728
    int size = 8; 

    int *A = (int*)malloc(size * sizeof(int));
    int *B = (int*)malloc(size * sizeof(int));
    int *C = (int*)malloc(size * 2 * sizeof(int)); 

	//start array clock
	clock_t startRNG, endRNG;
	startRNG = clock();
	
    //generateArray(A, size);
	A[0] = 1741389964; A[1] = 1166215685; A[2] = 656871634; A[3] = -1030419403;
	A[4] = -1042311416; A[5] = -1406574630; A[6] = -1602934049; A[7] = -1688181510;
    print_array("Array A", A, size);
    //generateArray(B, size);
	B[0] = 2028467558; B[1] = 1652238394; B[2] = 1079169700; B[3] = 754141167;
	B[4] = 570142334; B[5] = -136836139; B[6] = -551441583; B[7] = -969823145;
    print_array("Array B", B, size);
	
	//end array clock and printout of time
	endRNG = clock();
	double millisecondsRNG = static_cast<double>(endRNG - startRNG) / (CLOCKS_PER_SEC / 1000.0);
	std::cout << "\nTime to generate array 1 ms " << std::endl;

	//start clock 
	cudaEvent_t startGPU, stopGPU;
	
	// FLiMS
	cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
	float millisecondsGPU = 0;

	cudaEventRecord(startGPU);

	//declarations 
    int *d_A, *d_B, *d_C, *d_UsedA, *d_UsedB ;
    cudaMalloc(&d_A, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_C, size * 2 * sizeof(int));
    cudaMalloc(&d_UsedA, size * sizeof(int));
    cudaMalloc(&d_UsedB, size * sizeof(int));
    cudaMemcpy(d_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_UsedA, 0, size * sizeof(int));
    cudaMemset(d_UsedB, 0, size * sizeof(int));


	//insert main here
	
	int *d_a_indices4, *d_b_indices4;
	cudaMalloc(&d_a_indices4, 4 * sizeof(int));
	cudaMalloc(&d_b_indices4, 4 * sizeof(int));
	int a_indices4[4] = { 0, 1, 2, 3};
	int b_indices4[4] = { 3, 2, 1, 0};
	cudaMemcpy(d_a_indices4, a_indices4, 4 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_indices4, b_indices4, 4 * sizeof(int), cudaMemcpyHostToDevice);
	
	int elementsPerBlock = 4;
	int numBlocks = (size * 2 + elementsPerBlock - 1) / elementsPerBlock; 
	
	max_unit_kernel_AB_4<<<1, elementsPerBlock>>>(d_A, d_B, d_C, d_UsedA, d_UsedB, d_a_indices4, d_b_indices4, size);	
	
	cudaMemcpy(C, d_C, size * 2 * sizeof(int), cudaMemcpyDeviceToHost);
	
	print_array("unsorted C:", C, size*2);
	
	
	
	cas_kernel_4<<<numBlocks, elementsPerBlock>>>(d_C, size*2);  
	cudaMemcpy(C, d_C, size * 2 * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stopGPU);
	
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
	
	print_array("sorted C:", C, size*2);
	
	printf("Array length: %d\n", size * 2);
	
	bool sorted = isSorted(C, size * 2);
    if (sorted) 
	{
        printf("C is sorted in decreasing order. Test Success\n");
    } 
	else 
	{
        printf("C is NOT sorted in decreasing order. Test Failed\n");
    }
	
	int zeroCount = countZeros(C, size * 2);
	if (zeroCount == 0)
	{
	printf("0 not detected , Test Success\n");
	}
	else 
	{
	printf("\n0 detected , Test Failed\n");
	}
	
	int dupeCount = countDupes(C, size * 2);
	printf("C contains %d dupes value(s).\n", dupeCount);
	
	//printf("element value %d \n", elementsPerBlock);
	std::cout << "GPU Time: " << millisecondsGPU << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_UsedA);
    cudaFree(d_UsedB);
    free(A);
    free(B);
    free(C);

    return 0;
}







