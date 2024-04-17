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



#include <thrust/merge.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>



void generateAndSortArray(int* array, int size) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(INT_MIN, INT_MAX);

    //fill array with random numbers
    for (int i = 0; i < size; ++i) 
    {
        array[i] = dist(gen);
    }

    std::sort(array, array + size, std::greater<int>());
}

void print_array(const char* name, int* array, size_t size) 
{
    printf("%s: ", name);
    for (size_t i = 0; i < size; ++i) 
	{
        printf("%d ", array[i]);
    }
    printf("\n");
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
			//printf("Duplicate found: %d\n", arr[i]);
            count++;
        }
    }
    return count;
}



int main() {
	//65536 16384 32768 16777216 1048576 131072 262144 524288 1048576 2097152 4194304 8388608 , 33554432 , 67108864 , 134217728
	
    int size = 134217728; 

    int *A = (int*)malloc(size * sizeof(int));
    int *B = (int*)malloc(size * sizeof(int));
    int *C_thrust = (int*)malloc(size * 2 * sizeof(int)); 

	//start array clock
	clock_t startRNG, endRNG;
	startRNG = clock();
	
    generateAndSortArray(A, size);
    //print_array("Array A", A, size);
    generateAndSortArray(B, size);
    //print_array("Array B", B, size);
	
	//end array clock and printout of time
	endRNG = clock();
	double millisecondsRNG = static_cast<double>(endRNG - startRNG) / (CLOCKS_PER_SEC / 1000.0);
	std::cout << "Time to generate array " << millisecondsRNG << " ms" << std::endl;
	
	
	

	//start clock 
	cudaEvent_t startGPU, stopGPU;
	cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
	float thrustmillisecondsGPU = 0;

	
	
	thrust::device_vector<int> d_A_thrust(A, A + size);
    thrust::device_vector<int> d_B_thrust(B, B + size);
    thrust::device_vector<int> d_C_thrust(size * 2);
	
	cudaEventRecord(startGPU);
	thrust::merge(d_A_thrust.begin(), d_A_thrust.end(), d_B_thrust.begin(), d_B_thrust.end(), d_C_thrust.begin(), thrust::greater<int>());
	cudaEventRecord(stopGPU);
	
	thrust::copy(d_C_thrust.begin(), d_C_thrust.end(), C_thrust); 
	
	//cudaEventRecord(stopGPU);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&thrustmillisecondsGPU, startGPU, stopGPU);
	
	
	/*
	printf("sorted C_thrust: ");
    for (int i = 0; i < size * 2; i++) {
        printf("%d ", C_thrust[i]);
    }
    printf("\n");
	*/
	
	
	
	bool sorted = isSorted(C_thrust, size * 2);
    if (sorted) 
	{
        printf("The combined and processed array C_thrust is sorted in decreasing order.\n");
    } else 
	{
        printf("The combined and processed array C_thrust is NOT sorted in decreasing order.\n");
    }
	
	printf("Array length: %d\n", size * 2);
	
	
	int zeroCount = countZeros(C_thrust, size * 2);
	printf("The combined and processed array C contains %d occurrence(s) of the value 0.\n", zeroCount);

	int duplicateCount = countDupes(C_thrust, size * 2);
	printf("The combined and processed array C contains %d duplicate value(s).\n", duplicateCount);

	
	
	
	std::cout << "GPU Time: " << thrustmillisecondsGPU << " ms" << std::endl;

    return 0;
}