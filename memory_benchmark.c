#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

void tick(struct timeval *t)
{
    gettimeofday(t, NULL);
}

double tock(struct timeval *t)
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec - t->tv_sec) +
           ((double)(now.tv_usec - t->tv_usec) / 1000000);
}
#define vector_size 100000000

float *vectorA, *vectorB, *vectorC;
void loop_unrolling_add(int size)
{
    {
// #pragma ivdep
// #pragma omp simd
#pragma omp parallel for simd
        for (int i = 0; i < size / 4; i++)
        {
            vectorB[i] += vectorA[i];
            vectorB[4 * i] += vectorA[4 * i];
            vectorB[4 * i + 1] += vectorA[4 * i + 1];
            vectorB[4 * i + 2] += vectorA[4 * i + 2];
            vectorB[4 * i + 3] += vectorA[4 * i + 3];
        
        }
    }
}
void parallel_add(int size)
{
    {
#pragma omp parallel for simd
        for (int i = 0; i < size; i++)
        {
            vectorB[i] += vectorA[i];
        }
    }
}
void RandomVector(int n, float *A)
{
    for (int i = 0; i < n; ++i)
    {
        A[i] = (float)rand() / RAND_MAX;
    }
}
void direct_add(int size)
{
    for (int i = 0; i < size; i++)
    {
        vectorB[i] += vectorA[i];
    }
}

int main()
{
    // initialise A , B , C
    srand(time(NULL));

    struct timeval t;
    long long int size = 10000000;
    vectorA = (float *)malloc(size * sizeof(float));
    vectorB = (float *)malloc(size * sizeof(float));
    vectorC = (float *)malloc(size * sizeof(float));

    long long int num_operations = vector_size;
    long long int bytes = 2 * vector_size * sizeof(float);
    tick(&t);
    direct_add(size);
    double direct_add_time = tock(&t);
    printf("direct add bw: %f\n", (double)bytes / (direct_add_time * 1000000000));
    tick(&t);
    parallel_add(size);
    double parallel_add_time = tock(&t);
    printf("parallel add bw: %f\n", (double)bytes / (parallel_add_time * 1000000000));

    RandomVector(size, vectorA);
    RandomVector(size, vectorB);
    RandomVector(size, vectorC);
    tick(&t);
    loop_unrolling_add(size);
    double loop_unrolling_add_time = tock(&t);
    printf("loop unrolling add time %f\n", (double)bytes / (loop_unrolling_add_time * 1000000000));
    free(vectorA);
    free(vectorB);
    free(vectorC);

    return 0;

}