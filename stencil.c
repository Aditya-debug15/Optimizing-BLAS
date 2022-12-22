#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define _XOPEN_SOURCE

void tick(struct timeval *t)
{
    gettimeofday(t, NULL);
}

double tock(struct timeval *t)
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec - t->tv_sec) +
           ((double)(now.tv_usec - t->tv_usec) / 1000000.);
}

enum ImageType
{
    HD,
    UHD
};

void RandomMatrixImage(int h, int w, int k, float *A)
{
    int next_row = w + k - 1;
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; ++i)
        {
            A[j*(next_row) +i] = (float) rand()/RAND_MAX;
            // A[j * (next_row) + i] = (i + j);
        }
        for (int i = w; i < w + k - 1; i++)
        {
            A[j * (next_row) + i] = 0;
        }
    }
    for (int j = h; j < h + k - 1; j++)
    {
        for (int i = 0; i < w + k - 1; ++i)
        {
            A[j * next_row + i] = 0;
        }
    }
}

void RandomVector(int n, float *A)
{
    for (int i = 0; i < n; i++)
    {
        A[i] = (float) rand()/RAND_MAX;
        // A[i] = (1);
    }
}

int main(int argc, char *argv[])
{
    printf("Enter the image type (HD or UHD): \n");
    printf("h FOR HD\n");
    printf("u FOR UHD\n");
    char A;
    int k, w, h;
    struct timeval t0;
    scanf("%c", &A);
    for (int i = 0; i < 7; i++)
    {
        if (A == 'h')
        {
            k = i+3;
            w = 1920;
            h = 1080;
        }
        else if (A == 'u')
        {
            k = i+3;
            w = 3840;
            h = 2160;
        }
        else
        {
            printf("Wrong input\n");
            return 0;
        }
        int actual_w = w + k - 1;
        int actual_h = h + k - 1;
        float *X = (float *)malloc(actual_w * actual_h * sizeof(float));
        float *S = (float *)malloc(k * k * sizeof(float));
        float *Y = (float *)malloc(w * h * sizeof(float));
        // float X[actual_h * actual_w];
        // float S[k * k];
        // float Y[h * w];
        RandomMatrixImage(h, w, k, X);
        // for (int i = 0; i < actual_h; i++)
        // {
        //     for (int j = 0; j < actual_w; j++)
        //     {
        //         printf("%f ", X[i * actual_w + j]);
        //     }
        //     printf("\n");
        // }
        RandomVector(k * k, S);
        // RandomVector(k, S);
        tick(&t0);
        // stencil(X, HD, k, S, Y);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                Y[i*w+j] = 0.0;
                // #pragma omp simd
                for (int m = 0; m < k; m++)
                {
                    for (int n = 0; n < k; n++)
                    {
                       Y[i*w +j] += X[(i + m) * (actual_w) + j + n] * S[m * k + n];
                    }
                }
                // Y[i * w + j] = temp;
            }
        }
        // for (int i = 0; i < h; i++)
        // {
        //     for (int j = 0; j < w; j++)
        //     {
        //         printf("%f ", Y[i * w + j]);
        //     }
        //     printf("\n");
        // }
        double calctime = tock(&t0);
        double gflops = h*w*k*k*2 * 1e-09;
        double mem_bw = (h*w*k*k*8 * 1e-09) / calctime;
        printf("%d %f\n", k,gflops/calctime);
        // printf("Time (in milli-secs) %f\n", calctime * 1000);
        // printf("Memory Bandwidth (in GBytes/s): %f\n", (gflops/calctime) *4);
        // printf("Compute Throughput (in GFlops/s): %f\n", gflops / calctime);
        free(X);
        free(S);
        free(Y);
    }
    return 0;
}