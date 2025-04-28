#include<iostream>
#include<stdlib.h>
#include<omp.h>

using namespace std;

// Function declarations
void mergesort(int a[], int i, int j);
void merge(int a[], int i1, int j1, int i2, int j2);

// Merge Sort function (recursive)
void mergesort(int a[], int i, int j) {
    int mid;
    
    if (i < j) {
        mid = (i + j) / 2;
        
        // Parallelize the sorting of left and right halves
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                mergesort(a, i, mid);
            }

            #pragma omp section
            {
                mergesort(a, mid + 1, j);
            }
        }

        // Merge the two sorted halves
        merge(a, i, mid, mid + 1, j);
    }
}

// Merge two sorted subarrays
void merge(int a[], int i1, int j1, int i2, int j2) {
    int temp[1000];    // Temporary array for merging
    int i = i1, j = i2, k = 0;
    
    // Merge the two subarrays into temp[]
    while (i <= j1 && j <= j2) {
        if (a[i] < a[j]) {
            temp[k++] = a[i++];
        } else {
            temp[k++] = a[j++];
        }
    }

    // Copy remaining elements from both subarrays into temp[]
    while (i <= j1) temp[k++] = a[i++];
    while (j <= j2) temp[k++] = a[j++];

    // Copy merged data back into the original array
    for (i = i1, j = 0; i <= j2; i++, j++) {
        a[i] = temp[j];
    }
}

int main() {
    int *a, n, i;
    
    // Input the number of elements
    cout << "\nEnter total number of elements: ";
    cin >> n;
    
    a = new int[n];  // Dynamically allocate memory for the array

    // Input array elements
    cout << "\nEnter elements: ";
    for (i = 0; i < n; i++) {
        cin >> a[i];
    }

    // Start time for sorting
    double start_time = omp_get_wtime();
    
    // Sort the array using Merge Sort
    mergesort(a, 0, n - 1);
    
    // End time for sorting
    double end_time = omp_get_wtime();

    // Output sorted array
    cout << "\nSorted array: ";
    for (i = 0; i < n; i++) {
        cout << a[i] << " ";
    }
    cout << endl;

    // Output time taken to sort
    cout << "\nTime taken: " << (end_time - start_time) << " seconds." << endl;

    delete[] a;  // Free dynamically allocated memory

    return 0;
}
