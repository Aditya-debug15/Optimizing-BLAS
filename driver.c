#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "helper.h"

#define DEFAULT_VECTOR_LENGTH 100000000 // million
// computing Y = aX + Y
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Incorrect Input \n");
        exit(1);
    }
    struct timeval calc;
    double calctime;
    int n, m, k; // vector size
    double gflops;
    float *X, *Y;
    double *X1, *Y1;
    float *A, *B, *C;
    double *A1, *B1;
    // #pragma omp parallel
    if (strcmp(argv[1], "saxpy") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 1e6 + i * 10000;
            double a = (rand() % 10) * 0.076; // scalar value
            // Changes as per the problem
            gflops = 2.0 * n * 1e-09;

            X = (float *)malloc(n * sizeof(float));
            Y = (float *)malloc(n * sizeof(float));

            srand((unsigned)time(NULL));

            RandomVector(n, X);
            RandomVector(n, Y);

            tick(&calc);

            cblas_saxpy(n, a, X, 1, Y, 1);

            calctime = tock(&calc);

            // Changes as per the problem
            float mem_bw = 8.0 * n * 1e-09 / calctime;
            printf("%d %f\n", n, gflops / calctime);
            // printf("Time (in milli-secs) %f\n", calctime * 1000);
            // printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            // printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            free(X);
            free(Y);
        }
    }
    else if (strcmp(argv[1], "daxpy") == 0)
    {
        n = DEFAULT_VECTOR_LENGTH;
        double a = (rand() % 10) * 0.076; // scalar value
        // Changes as per the problem
        gflops = 2.0 * n * 1e-09;

        X1 = (double *)malloc(n * sizeof(double));
        Y1 = (double *)malloc(n * sizeof(double));

        srand((unsigned)time(NULL));

        RandomVector2(n, X1);
        RandomVector2(n, Y1);

        tick(&calc);

        cblas_daxpy(n, a, X1, 1, Y1, 1);

        calctime = tock(&calc);

        // Changes as per the problem
        float mem_bw = 16.0 * n * 1e-09 / calctime;
        printf("Time (in milli-secs) %f\n", calctime * 1000);
        printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
        printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
        free(X1);
        free(Y1);
    }
    else if (strcmp(argv[1], "sscal") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            float a = (rand() % 10) * 0.076; // scalar value
            n = 1e6 + i * 10000;
            // Changes as per the problem
            gflops = 1.0 * n * 1e-09;

            X = (float *)malloc(n * sizeof(float));

            srand((unsigned)time(NULL));

            RandomVector(n, X);

            tick(&calc);

            cblas_sscal(n, a, X, 1);

            calctime = tock(&calc);

            // Changes as per the problem
            float mem_bw = 4.0 * n * 1e-09 / calctime;
            printf("%d %f\n", n, gflops / calctime);
            // printf("Time (in milli-secs) %f\n", calctime * 1000);
            // printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            // printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            free(X);
        }
    }
    else if (strcmp(argv[1], "dscal") == 0)
    {
        double a = (rand() % 10) * 0.076; // scalar value
        n = DEFAULT_VECTOR_LENGTH;
        // Changes as per the problem
        gflops = 1.0 * n * 1e-09;

        X1 = (double *)malloc(n * sizeof(double));

        srand((unsigned)time(NULL));

        RandomVector2(n, X1);

        tick(&calc);

        cblas_dscal(n, a, X1, 1);

        calctime = tock(&calc);

        // Changes as per the problem
        float mem_bw = 8.0 * n * 1e-09 / calctime;
        printf("Time (in milli-secs) %f\n", calctime * 1000);
        printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
        printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
        free(X1);
    }
    else if (strcmp(argv[1], "sgemy") == 0)
    {
        n = 1e3;
        m = 1e5;
        float a = (rand() % 10) * 0.076; // scalar value
        float b = (rand() % 10) * 0.076; // scalar value
        // Changes as per the problem
        gflops = ((2.0 * n + 2) * m) * 1e-09;
        X = (float *)malloc(n * sizeof(float));
        Y = (float *)malloc(m * sizeof(float));
        A = (float *)malloc(n * m * sizeof(float));
        srand((unsigned)time(NULL));
        RandomVector(n, X);
        RandomVector(m, Y);
        RandomVector(n * m, A);
        tick(&calc);

        cblas_sgemv(101, 111, m, n, a, A, n, X, 1, b, Y, 1);

        calctime = tock(&calc);

        // Changes as per the problem
        float mem_bw = (4.0 * (n * m + n + m)) * 1e-09 / calctime;
        // printf("%d %f\n", n, gflops / calctime);
        printf("Time (in milli-secs) %f\n", calctime * 1000);
        printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
        printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
        free(X);
        free(Y);
        free(A);
    }
    else if (strcmp(argv[1], "sgemyt") == 0)
    {
        n = 1e5;
        m = 1e3;
        float a = (rand() % 10) * 0.076; // scalar value
        float b = (rand() % 10) * 0.076; // scalar value
        // float a = 1.0;
        // float b = 1.0;
        // Changes as per the problem
        gflops = ((2.0 * m + 2) * n) * 1e-09;
        X = (float *)malloc(m * sizeof(float));
        Y = (float *)malloc(n * sizeof(float));
        A = (float *)malloc(n * m * sizeof(float));
        srand((unsigned)time(NULL));
        RandomVector(m, X);
        RandomVector(n, Y);
        RandomVector((n * m), A);
        // A[0] = 1.0;
        // A[1] = 2.0;
        // A[2] = 3.0;
        // A[3] = 4.0;
        // A[4] = 5.0;
        // A[5] = 6.0;
        // X[0] = 1.0;
        // X[1] = 2.0;
        // Y[0] = 1.0;
        // Y[1] = 2.0;
        // Y[2] = 3.0;
        tick(&calc);
        cblas_sgemv(101, 112, m, n, a, A, n, X, 1, b, Y, 1);
        calctime = tock(&calc);
        // Changes as per the problem
        float mem_bw = (4.0 * (n * m + n + m)) * 1e-09 / calctime;
        // printf("%d %f\n", n, gflops / calctime);
        printf("Time (in milli-secs) %f\n", calctime * 1000);
        printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
        printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
        // for (int i = 0; i < n; i++)
        // {
        //     printf("%f ", Y[i]);
        // }
        // printf("\n");
        free(X);
        free(Y);
        free(A);
    }
    else if (strcmp(argv[1], "sgemyc") == 0)
    {
        n = 1000;
        m = 100000;
        float a = (rand() % 10) * 0.076; // scalar value
        float b = (rand() % 10) * 0.076; // scalar value
        // float a = 1.0;
        // float b = 0.0;
        // Changes as per the problem
        gflops = ((2.0 * n + 2) * m) * 1e-09;
        X = (float *)malloc(n * sizeof(float));
        Y = (float *)malloc(m * sizeof(float));
        A = (float *)malloc(n * m * sizeof(float));
        srand((unsigned)time(NULL));
        RandomVector(n, X);
        RandomVector(m, Y);
        RandomVector((n * m), A);
        tick(&calc);
        cblas_sgemv(102, 111, m, n, a, A, m, X, 1, b, Y, 1);
        calctime = tock(&calc);
        // Changes as per the problem
        float mem_bw = (4.0 * (n * m + n + m)) * 1e-09 / calctime;
        printf("Time (in milli-secs) %f\n", calctime * 1000);
        printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
        printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
        // for(int i=0;i<m;i++)
        // {
        //     printf("%f ",Y[i]);
        // }
        // printf("\n");
        // for(int i=0;i<n;i++)
        // {
        //     printf("%f ",X[i]);
        // }
        // printf("\n");
        // for(int i=0;i<n*m;i++)
        // {
        //     printf("%f ",A[i]);
        // }
        // printf("\n");
        free(X);
        free(Y);
        free(A);
    }
    else if (strcmp(argv[1], "sgemyct") == 0)
    {
        n = 100000;
        m = 1000;
        float a = (rand() % 10) * 0.076; // scalar value
        float b = (rand() % 10) * 0.076; // scalar value
        // Changes as per the problem
        gflops = ((2.0 * m + 2) * n * n) * 1e-09;
        X = (float *)malloc(m * sizeof(float));
        Y = (float *)malloc(n * sizeof(float));
        A = (float *)malloc(n * m * sizeof(float));
        srand((unsigned)time(NULL));
        RandomVector(m, X);
        RandomVector(n, Y);
        RandomVector((n * m), A);
        tick(&calc);
        cblas_sgemv(102, 112, m, n, a, A, m, X, 1, b, Y, 1);
        calctime = tock(&calc);
        // Changes as per the problem
        float mem_bw = (4.0 * (n * m + n + m)) * 1e-09 / calctime;
        printf("Time (in milli-secs) %f\n", calctime * 1000);
        printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
        printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
        // for(int i=0;i<n;i+=10000)
        // {
        //     printf("%f\n",Y[i]);
        // }x
        free(X);
        free(Y);
        free(A);
    }
    else if (strcmp(argv[1], "sgemm") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 300 + 10 * i;
            m = 300 + 10 * i;
            k = 300 + 10 * i;
            float a = (rand() % 10) * 0.076; // scalar value
            float b = (rand() % 10) * 0.076; // scalar value
            // float a = 1.0;
            // float b = 1.0;
            // Changes as per the problem
            gflops = ((2.0 * m + 2) * n * n) * 1e-09;
            A = (float *)malloc(m * n * sizeof(float));
            B = (float *)malloc(n * k * sizeof(float));
            C = (float *)malloc(m * k * sizeof(float));
            srand((unsigned)time(NULL));
            RandomVector((m * n), A);
            RandomVector((n * k), B);
            RandomVector((m * k), C);
            // A[0] = 1.0;
            // A[1] = 2.0;
            // A[2] = 3.0;
            // A[3] = 4.0;
            // A[4] = 5.0;
            // A[5] = 6.0;
            // B[0] = 1.0;
            // B[1] = 4.0;
            // B[2] = 2.0;
            // B[3] = 5.0;
            // B[4] = 3.0;
            // B[5] = 6.0;
            // C[0] = 1.0;
            // C[1] = 1.0;
            // C[2] = 1.0;
            // C[3] = 1.0;

            tick(&calc);
            cblas_sgemm(101, 111, 111, m, n, k, a, A, n, B, k, b, C, k);
            calctime = tock(&calc);
            // Changes as per the problem
            float mem_bw = (12.0 * (n * m)) * 1e-09 / calctime;
            // printf("%d %f\n",n, gflops/calctime);
            printf("Time (in milli-secs) %f\n", calctime * 1000);
            printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            // for (int i = 0; i < m * k; i++)
            // {
            //     printf("%f ", C[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*k;i++)
            // {
            //     printf("%f ",A[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*k;i++)
            // {
            //     printf("%f ",B[i]);
            // }
            // printf("\n");
            free(A);
            free(B);
            free(C);
        }
    }
    else if (strcmp(argv[1], "sgemmbt") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 300 + 10 * i;
            m = 300 + 10 * i;
            k = 300 + 10 * i;
            float a = (rand() % 10) * 0.076; // scalar value
            float b = (rand() % 10) * 0.076; // scalar value
            // float a = 1.0;
            // float b = 1.0;
            // Changes as per the problem
            gflops = ((2.0 * m + 2) * n * n) * 1e-09;
            A = (float *)malloc(m * n * sizeof(float));
            B = (float *)malloc(n * k * sizeof(float));
            C = (float *)malloc(m * k * sizeof(float));
            srand((unsigned)time(NULL));
            RandomVector((m * n), A);
            RandomVector((n * k), B);
            RandomVector((m * k), C);
            // A[0] = 1.0;
            // A[1] = 2.0;
            // A[2] = 3.0;
            // A[3] = 4.0;
            // A[4] = 5.0;
            // A[5] = 6.0;
            // B[0] = 1.0;
            // B[1] = 4.0;
            // B[2] = 7.0;
            // B[3] = 5.0;
            // B[4] = 0.0;
            // B[5] = 6.0;
            // C[0] = 1.0;
            // C[1] = 1.0;
            // C[2] = 1.0;
            // C[3] = 1.0;
            tick(&calc);
            cblas_sgemm(101, 111, 112, m, n, k, a, A, n, B, k, b, C, k);
            calctime = tock(&calc);
            // Changes as per the problem
            float mem_bw = (12.0 * (n * m)) * 1e-09 / calctime;
            // printf("%d %f\n",n, gflops/calctime);
            printf("Time (in milli-secs) %f\n", calctime * 1000);
            printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            // for(int i=0;i<m*k;i++)
            // {
            //     printf("%f ",C[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*n;i++)
            // {
            //     printf("%f ",A[i]);
            // }
            // printf("\n");
            // for(int i=0;i<n*k;i++)
            // {
            //     printf("%f ",B[i]);
            // }
            // printf("\n");
            free(A);
            free(B);
            free(C);
        }
    }
    else if (strcmp(argv[1], "sgemmat") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 300 + 10 * i;
            m = 300 + 10 * i;
            k = 300 + 10 * i;
            float a = (rand() % 10) * 0.076; // scalar value
            float b = (rand() % 10) * 0.076; // scalar value
            // float a = 1.0;
            // float b = 1.0;
            // Changes as per the problem
            gflops = ((2.0 * m + 2) * n * n) * 1e-09;
            A = (float *)malloc(m * n * sizeof(float));
            B = (float *)malloc(n * k * sizeof(float));
            C = (float *)malloc(m * k * sizeof(float));
            srand((unsigned)time(NULL));
            RandomVector((m * n), A);
            RandomVector((n * k), B);
            RandomVector((m * k), C);
            // A[0] = 1.0;
            // A[1] = 3.0;
            // A[2] = 5.0;
            // A[3] = 7.0;
            // B[0] = 2.0;
            // B[1] = 4.0;
            // B[2] = 6.0;
            // B[3] = 8.0;
            // C[0] = 1.0;
            // C[1] = 1.0;
            // C[2] = 1.0;
            // C[3] = 1.0;
            tick(&calc);
            cblas_sgemm(101, 112, 111, m, n, k, a, A, n, B, k, b, C, k);
            calctime = tock(&calc);
            // Changes as per the problem
            float mem_bw = (12.0 * (n * m)) * 1e-09 / calctime;
            // printf("%d %f\n",n, gflops/calctime);
            printf("Time (in milli-secs) %f\n", calctime * 1000);
            printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            // for (int i = 0; i < m * k; i++)
            // {
            //     printf("%f ", C[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*n;i++)
            // {
            //     printf("%f ",A[i]);
            // }
            // printf("\n");
            // for(int i=0;i<n*k;i++)
            // {
            //     printf("%f ",B[i]);
            // }
            // printf("\n");
            free(A);
            free(B);
            free(C);
        }
    }
    else if (strcmp(argv[1], "sgemmatbt") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 300 + 10 * i;
            m = 300 + 10 * i;
            k = 300 + 10 * i;

            float a = (rand() % 10) * 0.076; // scalar value
            float b = (rand() % 10) * 0.076; // scalar value
            // float a = 1.0;
            // float b = 1.0;
            // Changes as per the problem
            gflops = ((2.0 * m + 2) * n * n) * 1e-09;
            A = (float *)malloc(m * n * sizeof(float));
            B = (float *)malloc(n * k * sizeof(float));
            C = (float *)malloc(m * k * sizeof(float));
            srand((unsigned)time(NULL));
            RandomVector((m * n), A);
            RandomVector((n * k), B);
            RandomVector((m * k), C);
            // A[0] = 1.0;
            // A[1] = 3.0;
            // A[2] = 5.0;
            // A[3] = 7.0;
            // B[0] = 2.0;
            // B[1] = 4.0;
            // B[2] = 6.0;
            // B[3] = 8.0;
            // C[0] = 1.0;
            // C[1] = 1.0;
            // C[2] = 1.0;
            // C[3] = 1.0;
            tick(&calc);
            cblas_sgemm(101, 112, 112, m, n, k, a, A, n, B, k, b, C, k);
            calctime = tock(&calc);
            // Changes as per the problem
            float mem_bw = (12.0 * (n * m)) * 1e-09 / calctime;
            // printf("%d %f\n",n, gflops/calctime);
            printf("Time (in milli-secs) %f\n", calctime * 1000);
            printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            // for (int i = 0; i < m * k; i++)
            // {
            //     printf("%f ", C[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*n;i++)
            // {
            //     printf("%f ",A[i]);
            // }
            // printf("\n");
            // for(int i=0;i<n*k;i++)
            // {
            //     printf("%f ",B[i]);
            // }
            // printf("\n");
            free(A);
            free(B);
            free(C);
        }
    }
    else if (strcmp(argv[1], "sgemmc") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 300 + 10 * i;
            m = 300 + 10 * i;
            k = 300 + 10 * i;

            float a = (rand() % 10) * 0.076; // scalar value
            float b = (rand() % 10) * 0.076; // scalar value
            // float a = 1.0;
            // float b = 1.0;
            // Changes as per the problem
            gflops = ((2.0 * m + 2) * n * n) * 1e-09;
            A = (float *)malloc(m * n * sizeof(float));
            B = (float *)malloc(n * k * sizeof(float));
            C = (float *)malloc(m * k * sizeof(float));
            srand((unsigned)time(NULL));
            RandomVector((m * n), A);
            RandomVector((n * k), B);
            RandomVector((m * k), C);
            // A[0] = 1.0;
            // A[1] = 3.0;
            // A[2] = 5.0;
            // A[3] = 7.0;
            // B[0] = 2.0;
            // B[1] = 4.0;
            // B[2] = 6.0;
            // B[3] = 8.0;
            // C[0] = 1.0;
            // C[1] = 1.0;
            // C[2] = 1.0;
            // C[3] = 1.0;
            tick(&calc);
            cblas_sgemm(102, 111, 111, m, n, k, a, A, n, B, k, b, C, k);
            calctime = tock(&calc);
            // Changes as per the problem
            float mem_bw = (12.0 * (n * m)) * 1e-09 / calctime;
            // printf("%d %f\n",n, gflops/calctime);
            printf("Time (in milli-secs) %f\n", calctime * 1000);
            printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            // for (int i = 0; i < m * k; i++)
            // {
            //     printf("%f ", C[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*n;i++)
            // {
            //     printf("%f ",A[i]);
            // }
            // printf("\n");
            // for(int i=0;i<n*k;i++)
            // {
            //     printf("%f ",B[i]);
            // }
            // printf("\n");
            free(A);
            free(B);
            free(C);
        }
    }
    else if (strcmp(argv[1], "sgemmcbt") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 300 + 10 * i;
            m = 300 + 10 * i;
            k = 300 + 10 * i;
            float a = (rand() % 10) * 0.076; // scalar value
            float b = (rand() % 10) * 0.076; // scalar value
            // float a = 1.0;
            // float b = 1.0;
            // Changes as per the problem
            gflops = ((2.0 * m + 2) * n * n) * 1e-09;
            A = (float *)malloc(m * n * sizeof(float));
            B = (float *)malloc(n * k * sizeof(float));
            C = (float *)malloc(m * k * sizeof(float));
            srand((unsigned)time(NULL));
            RandomVector((m * n), A);
            RandomVector((n * k), B);
            RandomVector((m * k), C);
            // A[0] = 1.0;
            // A[1] = 3.0;
            // A[2] = 5.0;
            // A[3] = 7.0;
            // B[0] = 2.0;
            // B[1] = 4.0;
            // B[2] = 6.0;
            // B[3] = 8.0;
            // C[0] = 1.0;
            // C[1] = 1.0;
            // C[2] = 1.0;
            // C[3] = 1.0;
            tick(&calc);
            cblas_sgemm(102, 111, 112, m, n, k, a, A, n, B, k, b, C, k);
            calctime = tock(&calc);
            // Changes as per the problem
            float mem_bw = (12.0 * (n * m)) * 1e-09 / calctime;
            // printf("%d %f\n",n, gflops/calctime);
            printf("Time (in milli-secs) %f\n", calctime * 1000);
            printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            // for (int i = 0; i < m * k; i++)
            // {
            //     printf("%f ", C[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*n;i++)
            // {
            //     printf("%f ",A[i]);
            // }
            // printf("\n");
            // for(int i=0;i<n*k;i++)
            // {
            //     printf("%f ",B[i]);
            // }
            // printf("\n");
            free(A);
            free(B);
            free(C);
        }
    }
    else if (strcmp(argv[1], "sgemmcat") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 300 + 10 * i;
            m = 300 + 10 * i;
            k = 300 + 10 * i;
            float a = (rand() % 10) * 0.076; // scalar value
            float b = (rand() % 10) * 0.076; // scalar value
            // float a = 1.0;
            // float b = 1.0;
            // Changes as per the problem
            gflops = ((2.0 * m + 2) * n * n) * 1e-09;
            A = (float *)malloc(m * n * sizeof(float));
            B = (float *)malloc(n * k * sizeof(float));
            C = (float *)malloc(m * k * sizeof(float));
            srand((unsigned)time(NULL));
            RandomVector((m * n), A);
            RandomVector((n * k), B);
            RandomVector((m * k), C);
            // A[0] = 1.0;
            // A[1] = 3.0;
            // A[2] = 5.0;
            // A[3] = 7.0;
            // B[0] = 2.0;
            // B[1] = 4.0;
            // B[2] = 6.0;
            // B[3] = 8.0;
            // C[0] = 1.0;
            // C[1] = 1.0;
            // C[2] = 1.0;
            // C[3] = 1.0;
            tick(&calc);
            cblas_sgemm(102, 112, 111, m, n, k, a, A, n, B, k, b, C, k);
            calctime = tock(&calc);
            // Changes as per the problem
            float mem_bw = (12.0 * (n * m)) * 1e-09 / calctime;
            // printf("%d %f\n",n, gflops/calctime);
            printf("Time (in milli-secs) %f\n", calctime * 1000);
            printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            // for (int i = 0; i < m * k; i++)
            // {
            //     printf("%f ", C[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*n;i++)
            // {
            //     printf("%f ",A[i]);
            // }
            // printf("\n");
            // for(int i=0;i<n*k;i++)
            // {
            //     printf("%f ",B[i]);
            // }
            // printf("\n");
            free(A);
            free(B);
            free(C);
        }
    }
    else if (strcmp(argv[1], "sgemmcatbt") == 0)
    {
        for (int i = 0; i < 10; i++)
        {
            n = 300 + 10 * i;
            m = 300 + 10 * i;
            k = 300 + 10 * i;
            float a = (rand() % 10) * 0.076; // scalar value
            float b = (rand() % 10) * 0.076; // scalar value
            // float a = 1.0;
            // float b = 1.0;
            // Changes as per the problem
            gflops = ((2.0 * m + 2) * n * n) * 1e-09;
            A = (float *)malloc(m * n * sizeof(float));
            B = (float *)malloc(n * k * sizeof(float));
            C = (float *)malloc(m * k * sizeof(float));
            srand((unsigned)time(NULL));
            RandomVector((m * n), A);
            RandomVector((n * k), B);
            RandomVector((m * k), C);
            // A[0] = 1.0;
            // A[1] = 3.0;
            // A[2] = 5.0;
            // A[3] = 7.0;
            // B[0] = 2.0;
            // B[1] = 4.0;
            // B[2] = 6.0;
            // B[3] = 8.0;
            // C[0] = 1.0;
            // C[1] = 1.0;
            // C[2] = 1.0;
            // C[3] = 1.0;
            tick(&calc);
            cblas_sgemm(102, 112, 112, m, n, k, a, A, n, B, k, b, C, k);
            calctime = tock(&calc);
            // Changes as per the problem
            float mem_bw = (12.0 * (n * m)) * 1e-09 / calctime;
            // printf("%d %f\n",n, gflops/calctime);

            printf("Time (in milli-secs) %f\n", calctime * 1000);
            printf("Memory Bandwidth (in GBytes/s): %f\n", mem_bw);
            printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
            // for (int i = 0; i < m * k; i++)
            // {
            //     printf("%f ", C[i]);
            // }
            // printf("\n");
            // for(int i=0;i<m*n;i++)
            // {
            //     printf("%f ",A[i]);
            // }
            // printf("\n");
            // for(int i=0;i<n*k;i++)
            // {
            //     printf("%f ",B[i]);
            // }
            // printf("\n");
            free(A);
            free(B);
            free(C);
        }
    }
}
