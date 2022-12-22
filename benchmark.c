#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 10000

int main()
{
    // declare a matrix of size 1000*1000 using malloc
    float **matrix = (float **)malloc(N * sizeof(float *));
    for (int i = 0; i < N; i++)
    {
        matrix[i] = (float *)malloc(N * sizeof(float));
    }
    // fill matrix with a random floating point number withing range [0,1000]
    int a;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a = rand() % 1000;
            matrix[i][j] = a*i*0.49956;
        }
    }
    // declare a vector of size 1000
    float *vector = (float *)malloc(N * sizeof(float));
    // fill vector with a random floating point number withing range [0,1000]
    for (int i = 0; i < N; i++)
    {
        a = rand() % 1000;
        vector[i] = a*i*0.48756;
    }
    // declare a vector of size 1000
    float *vector2 = (float *)malloc(N * sizeof(float));
    // fill vector with a random floating point number withing range [0,1000]
    for (int i = 0; i < N; i++)
    {
        a = rand() % 1000;
        vector2[i] = a*i*0.48756;
    }
    float alpha=0.3452,beta=0.5635;
    // calculate the vector2 = beta*vector2 + alpha*matrix*vector
    long startsec, finisec;
    startsec = time(0);
    for (int i = 0; i < N; i++)
    {
        vector2[i] = beta*vector2[i];
        float AX = matrix[i][0]*vector[0];
        for (int j = 1; j < N; j++)
        {
            AX += matrix[i][j]*vector[j];
        }
        AX *= alpha;
        vector2[i] += AX;
    }
    finisec = time(0);
    printf("time: %ld\n", startsec);
    printf("time: %ld\n", finisec);
}