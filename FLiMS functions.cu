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
    std::uniform_int_distribution<int> dist( , INT_MAX);

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


__global__ void max_unit_kernel_AB_1024(int *A, int *B, int *C, int *UsedA, int *UsedB, int *a_indices, int *b_indices, int size) 
{
    
	int i = threadIdx.x;
    //inside kernel loop
    for(int offset = 0; offset < size * 2; offset += 1024) 
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
                a_indices[i] = (a_index + 1024 < size) ? a_index + 1024 : a_index;
            } 
			else if (b_state == 1) 
			{
				//set c array to value
                C[i + offset] = B[b_index];
				//set used to 2 showing that its been used
                UsedB[b_index] += 1;
				//check to see if new index value is less than size , if so switch to it otherwise don't change
                b_indices[i] = (b_index + 1024 < size) ? b_index + 1024 : b_index;
            }
        }
    }
}

__global__ void cas_kernel_1024(int *C, int size) 
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int offset = blockId * blockDim.x;
    
    while (offset < size) 
	{
        __shared__ int temp[1024];
        int idx = threadIdx.x;

        //offset is within bounds 
        if (offset + idx < size) 
		{
            
            temp[idx] = C[offset + idx];
        }

        __syncthreads();


	//CAS with 512
    int ixj = idx ^ 512;
    if ((idx < ixj) && ((idx & 512) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();
	
	
	//CAS with 256
    ixj = idx ^ 256;
    if ((idx < ixj) && ((idx & 256) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();

    //CAS with 128
    ixj = idx ^ 128;
    if ((idx < ixj) && ((idx & 128) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();

    //CAS with 64
    ixj = idx ^ 64;
    if ((idx < ixj) && ((idx & 64) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();

	//CAS with 32
    ixj = idx ^ 32;
    if ((idx < ixj) && ((idx & 32) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();

    //CAS with 16
    ixj = idx ^ 16;
    if ((idx < ixj) && ((idx & 16) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();

    //CAS with 8
    ixj = idx ^ 8;
    if ((idx < ixj) && ((idx & 8) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();

    //CAS with 4
    ixj = idx ^ 4;
    if ((idx < ixj) && ((idx & 4) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();

    //CAS with 2
    ixj = idx ^ 2;
    if ((idx < ixj) && ((idx & 2) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
        int tmp = temp[idx];
        temp[idx] = temp[ixj];
        temp[ixj] = tmp;
    }

    __syncthreads();

    //CAS with 1
    ixj = idx ^ 1;
    if ((idx < ixj) && ((idx & 1) == 0 ? temp[idx] < temp[ixj] : temp[idx] > temp[ixj])) 
	{
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
	//65536 16384 32768 166777216 1048576 131072 262144 524288 1048576
    int size = 16777216; 

    int *A = (int*)malloc(size * sizeof(int));
    int *B = (int*)malloc(size * sizeof(int));
    int *C = (int*)malloc(size * 2 * sizeof(int)); 

	//start array clock
	clock_t startRNG, endRNG;
	startRNG = clock();
	
    generateArray(A, size);
    //print_array("Array A", A, size);
    generateArray(B, size);
    //print_array("Array B", B, size);
	
	//end array clock and printout of time
	endRNG = clock();
	double millisecondsRNG = static_cast<double>(endRNG - startRNG) / (CLOCKS_PER_SEC / 1000.0);
	std::cout << "\nTime to generate array " << millisecondsRNG << " ms" << std::endl;

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
	
	//1024 elements
	int *d_a_indices1024, *d_b_indices1024;
	cudaMalloc(&d_a_indices1024, 1024 * sizeof(int));
	cudaMalloc(&d_b_indices1024, 1024 * sizeof(int));
	int a_indices1024[1024] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023};
	int b_indices1024[1024] = {1023, 1022, 1021, 1020, 1019, 1018, 1017, 1016, 1015, 1014, 1013, 1012, 1011, 1010, 1009, 1008, 1007, 1006, 1005, 1004, 1003, 1002, 1001, 1000, 999, 998, 997, 996, 995, 994, 993, 992, 991, 990, 989, 988, 987, 986, 985, 984, 983, 982, 981, 980, 979, 978, 977, 976, 975, 974, 973, 972, 971, 970, 969, 968, 967, 966, 965, 964, 963, 962, 961, 960, 959, 958, 957, 956, 955, 954, 953, 952, 951, 950, 949, 948, 947, 946, 945, 944, 943, 942, 941, 940, 939, 938, 937, 936, 935, 934, 933, 932, 931, 930, 929, 928, 927, 926, 925, 924, 923, 922, 921, 920, 919, 918, 917, 916, 915, 914, 913, 912, 911, 910, 909, 908, 907, 906, 905, 904, 903, 902, 901, 900, 899, 898, 897, 896, 895, 894, 893, 892, 891, 890, 889, 888, 887, 886, 885, 884, 883, 882, 881, 880, 879, 878, 877, 876, 875, 874, 873, 872, 871, 870, 869, 868, 867, 866, 865, 864, 863, 862, 861, 860, 859, 858, 857, 856, 855, 854, 853, 852, 851, 850, 849, 848, 847, 846, 845, 844, 843, 842, 841, 840, 839, 838, 837, 836, 835, 834, 833, 832, 831, 830, 829, 828, 827, 826, 825, 824, 823, 822, 821, 820, 819, 818, 817, 816, 815, 814, 813, 812, 811, 810, 809, 808, 807, 806, 805, 804, 803, 802, 801, 800, 799, 798, 797, 796, 795, 794, 793, 792, 791, 790, 789, 788, 787, 786, 785, 784, 783, 782, 781, 780, 779, 778, 777, 776, 775, 774, 773, 772, 771, 770, 769, 768, 767, 766, 765, 764, 763, 762, 761, 760, 759, 758, 757, 756, 755, 754, 753, 752, 751, 750, 749, 748, 747, 746, 745, 744, 743, 742, 741, 740, 739, 738, 737, 736, 735, 734, 733, 732, 731, 730, 729, 728, 727, 726, 725, 724, 723, 722, 721, 720, 719, 718, 717, 716, 715, 714, 713, 712, 711, 710, 709, 708, 707, 706, 705, 704, 703, 702, 701, 700, 699, 698, 697, 696, 695, 694, 693, 692, 691, 690, 689, 688, 687, 686, 685, 684, 683, 682, 681, 680, 679, 678, 677, 676, 675, 674, 673, 672, 671, 670, 669, 668, 667, 666, 665, 664, 663, 662, 661, 660, 659, 658, 657, 656, 655, 654, 653, 652, 651, 650, 649, 648, 647, 646, 645, 644, 643, 642, 641, 640, 639, 638, 637, 636, 635, 634, 633, 632, 631, 630, 629, 628, 627, 626, 625, 624, 623, 622, 621, 620, 619, 618, 617, 616, 615, 614, 613, 612, 611, 610, 609, 608, 607, 606, 605, 604, 603, 602, 601, 600, 599, 598, 597, 596, 595, 594, 593, 592, 591, 590, 589, 588, 587, 586, 585, 584, 583, 582, 581, 580, 579, 578, 577, 576, 575, 574, 573, 572, 571, 570, 569, 568, 567, 566, 565, 564, 563, 562, 561, 560, 559, 558, 557, 556, 555, 554, 553, 552, 551, 550, 549, 548, 547, 546, 545, 544, 543, 542, 541, 540, 539, 538, 537, 536, 535, 534, 533, 532, 531, 530, 529, 528, 527, 526, 525, 524, 523, 522, 521, 520, 519, 518, 517, 516, 515, 514, 513, 512, 511, 510, 509, 508, 507, 506, 505, 504, 503, 502, 501, 500, 499, 498, 497, 496, 495, 494, 493, 492, 491, 490, 489, 488, 487, 486, 485, 484, 483, 482, 481, 480, 479, 478, 477, 476, 475, 474, 473, 472, 471, 470, 469, 468, 467, 466, 465, 464, 463, 462, 461, 460, 459, 458, 457, 456, 455, 454, 453, 452, 451, 450, 449, 448, 447, 446, 445, 444, 443, 442, 441, 440, 439, 438, 437, 436, 435, 434, 433, 432, 431, 430, 429, 428, 427, 426, 425, 424, 423, 422, 421, 420, 419, 418, 417, 416, 415, 414, 413, 412, 411, 410, 409, 408, 407, 406, 405, 404, 403, 402, 401, 400, 399, 398, 397, 396, 395, 394, 393, 392, 391, 390, 389, 388, 387, 386, 385, 384, 383, 382, 381, 380, 379, 378, 377, 376, 375, 374, 373, 372, 371, 370, 369, 368, 367, 366, 365, 364, 363, 362, 361, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351, 350, 349, 348, 347, 346, 345, 344, 343, 342, 341, 340, 339, 338, 337, 336, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310, 309, 308, 307, 306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296, 295, 294, 293, 292, 291, 290, 289, 288, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275, 274, 273, 272, 271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 256, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
	cudaMemcpy(d_a_indices1024, a_indices1024, 1024 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_indices1024, b_indices1024, 1024 * sizeof(int), cudaMemcpyHostToDevice);

	int elementsPerBlock = 1024;

	int numBlocks = (size * 2 + elementsPerBlock - 1) / elementsPerBlock;
	max_unit_kernel_AB_1024<<<1, elementsPerBlock>>>(d_A, d_B, d_C, d_UsedA, d_UsedB, d_a_indices1024, d_b_indices1024, size);	
	cas_kernel_1024<<<numBlocks, elementsPerBlock>>>(d_C, size*2);  
	cudaMemcpy(C, d_C, size * 2 * sizeof(int), cudaMemcpyDeviceToHost);
	
	
	cudaEventRecord(stopGPU);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
	
	//print_array("sorted C:", C, size*2);
	
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







