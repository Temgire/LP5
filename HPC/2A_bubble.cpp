#include <iostream>
#include <omp.h>
using namespace std;

void parallelBubbleSort(int arr[], int n) {
    bool swapped = true;

    for (int i = 0; i < n && swapped; i++) {
        swapped = false;

        // EVEN phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        // ODD phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
    }
}

int main() {
    int arr[] = {5, 3, 8, 4, 2};
    int n = sizeof(arr) / sizeof(arr[0]);

    cout << "Initial array: ";
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << endl;

    parallelBubbleSort(arr, n);

    cout << "Sorted array: ";
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << endl;

    return 0;
}
