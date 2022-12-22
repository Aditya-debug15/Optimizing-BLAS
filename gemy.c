#include<stdio.h>
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

void cblas_sscal(const int N, const float alpha, float *X, const int incX)
{
#pragma omp parallel
    {
    #pragma omp for simd
        for (int i = 0; i < N; ++i)
        {
            X[i] = alpha * X[i];
        }
    }        
}
void cblas_dscal(const int N, const double alpha, double *X, const int incX)
{
#pragma omp simd
    for (int i = 0; i < N; i++)
    {
        X[i] = alpha * X[i * incX];
    }
}

void cblas_saxpy(const int N, const float alpha, const float *X,const int incX, float *Y, const int incY)
{
#pragma omp parallel
    {
    #pragma omp for simd
        for (int i = 0; i < N; ++i)
        {
            Y[i*incY] = alpha * X[i*incX] + Y[i*incY];
        }
    }
}
void cblas_daxpy(const int N, const double alpha, const double *X,const int incX, double *Y, const int incY)
{
#pragma omp parallel
    {
    #pragma omp for simd
        for (int i = 0; i < N; ++i)
        {
            Y[i*incY] = alpha * X[i*incX] + Y[i*incY];
        }
    }
}

float cblas_sdot (const int N, const float *X, const int incX, const float *Y, const int incY)
{
    float dpsum = 0;
    #pragma omp simd
    for(int i = 0; i < N; ++i)
    {
        dpsum += (X[i*incX]*Y[i*incY]);
    }
    return dpsum;
}


void cblas_sgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha, const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY)
{
    if (order == CblasRowMajor)
    {
        if (TransA == CblasNoTrans)
        {
            #pragma omp parallel for simd
            for (int i = 0; i < M; i++)
            {
                float AX = cblas_sdot(N,A+i*lda,1,X,incX);
                Y[i*incY] = (beta * Y[i*incY]) + (alpha * AX);
            }
        }
        else{
            // // // #pragma omp parallel for simd
            // for (int j = 0; j < N; j++)
            // {
            //     float AX = 0;
            //     // #pragma omp simd
            //     for (int i = 0; i < M-1; i++)
            //     {
            //         // __builtin_prefetch(A+(i+1)*lda+j);
            //         AX += A[i*lda+j] * X[i*incX];
            //     }
            //     AX += A[(M-1)*lda+j] * X[(M-1)*incX];
            //     Y[j*incY] = (beta * Y[j*incY]) + (alpha * AX);
            // }
            if(M>N)
            {
                cblas_sscal(N,beta,Y,incY);
                #pragma omp parallel for simd
                for(int i = 0; i < M; i++)
                {
                    cblas_saxpy(N,alpha*X[i*incX],A+(i*lda),1,Y,incY);
                }
            }
            else{
                cblas_sscal(N,beta,Y,incY);
                for(int i = 0; i < M; i++)
                {
                    cblas_saxpy(N,alpha*X[i*incX],A+(i*lda),1,Y,incY);
                }
            }

        }
    } 
    else{
        if(TransA == CblasNoTrans)
        {
            // for(int i = 0; i < M; i++)
            // {
            //     float AX = 0;
            //     // #pragma omp simd
            //     for (int j = 0; j < N-1; j++)
            //     {
            //         // __builtin_prefetch(A+i*lda+j+1);
            //         AX += A[i*lda+j] * X[j*incX];
            //     }
            //     AX += A[i*lda+N-1] * X[(N-1)*incX];
            //     Y[i*incY] = (beta * Y[i*incY]) + (alpha * AX);
            // }

            cblas_sscal(M,beta,Y,incY);
            #pragma omp parallel for simd
            for(int i=0;i<N;i++)
            {
                cblas_saxpy(M,alpha*X[i*incX],A+(i*lda),1,Y,incY);
            }
        }
        else{
            #pragma omp parallel for simd
            for (int i=0;i<N;i++)
            {
                float AX = cblas_sdot(M,A+i*lda,1,X,incX);
                Y[i*incY] = (beta * Y[i*incY]) + (alpha * AX);
            }
        }
    }
}
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc)
{
    if(Order == CblasRowMajor)
    {
        if(TransA == CblasNoTrans)
        {
            if(TransB == CblasNoTrans)
            {
                for(int i=0;i<M;i++)
                {
                    cblas_sgemv(CblasRowMajor,CblasTrans,N,K,alpha,B,ldb,A+i*lda,1,beta,C+i*ldc,1);
                }
            }
            else
            {
                #pragma omp for simd
                for(int i=0;i<M;i++)
                {
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,K,N,alpha,B,N,A+i*lda,1,beta,C+i*ldc,1);
                }
            }
        }
        else{
            if(TransB == CblasNoTrans)
            {
                for(int i=0;i<M;i++)
                {
                    cblas_sgemv(CblasRowMajor,CblasTrans,N,K,alpha,B,ldb,A+i,lda,beta,C+i*ldc,1);
                }
            }
            else{
                #pragma omp parallel for simd
                for(int i=0;i<M;i++)
                {
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,K,N,alpha,B,N,A+i,lda,beta,C+i*ldc,1);
                }
            }
        }
    }
    else{
        if(TransA == CblasNoTrans)
        {
            if(TransB == CblasNoTrans)
            {
                #pragma omp parallel for simd
                for(int i=0;i<M;i++)
                {
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,K,N,alpha,B,N,A+i,lda,beta,C+i*ldc,1);
                }
            }
            else{
                for(int i=0;i<M;i++)
                {
                    cblas_sgemv(CblasRowMajor,CblasTrans,N,K,alpha,B,ldb,A+i,lda,beta,C+i*ldc,1);
                }
            }
        }
        else{
            if(TransB == CblasNoTrans)
            {
                #pragma omp parallel for simd
                for(int i=0;i<M;i++)
                {
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,K,N,alpha,B,N,A+i*lda,1,beta,C+i*ldc,1);
                }
            }
            else{
                for(int i=0;i<M;i++)
                {
                    cblas_sgemv(CblasRowMajor,CblasTrans,N,K,alpha,B,ldb,A+i*lda,1,beta,C+i*ldc,1);
                }
            }
        }
    }
}