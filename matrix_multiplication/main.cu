#include "stdio.h"
#include <time.h>
#include <sys/time.h>
typedef int DTYPE;

void matrix_multiplication_serial_1(DTYPE* a, DTYPE* b, DTYPE* c, int m, int n, int l)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            DTYPE temp = 0;
            for(int k = 0; k < l; k++)
            {
                temp += a[i*l+k] * b[k*n+j];
            }
            c[i*n+j] = temp;
        }
    }
}

void matrix_multiplication_serial_2(DTYPE* a, DTYPE* b, DTYPE* c, int m, int n, int l)
{
    //init c
    for(int row = 0; row < m; ++row)
    {
        for(int col = 0; col < n; ++col)
        {
            c[col + row * n] = 0;
        }
    }
    for(int i = 0; i < m; i++)
    {
        for(int k = 0; k < l; k++)
        {
            for(int j = 0; j < n; j++)
            {
                c[i*n+j] += a[i*l+k] * b[k*n+j];
            }
        }
    }
}

void matrix_multiplication_serial_3(DTYPE* a, DTYPE* b, DTYPE* c, int m, int n, int l)
{
    //transform b
    DTYPE* b1 = new DTYPE[n * l];
    for(int row = 0; row < l; ++row)
    {
        for(int col = 0; col < n; ++col)
        {
            b1[col + row * n] = b[col + row * n];
        }
    }
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            c[i*n+j] = 0;
            for(int k = 0; k < l; k++)
            {
                c[i*n+j] += a[i*l+k] * b[k*n+j];
            }
        }
    }
}

int main()
{
    int m = 10;
    int l = 20;
    int n = 30;
    // a:m*l, b:l*n, c:m*n
    DTYPE *a = new DTYPE[m * l];
    DTYPE *b = new DTYPE[l * n];
    DTYPE *c = new DTYPE[m * n];
    // init a
    printf("a:\n");
    for(int row = 0; row < m; ++row)
    {
        for(int col = 0; col < l; ++col)
        {
            a[col + row * l] = (col + row) % 256;
            printf("%3d ",a[col + row * l]);
        }
        printf("\n");
    }
    //init b
    printf("b:\n");
    for(int row = 0; row < l; ++row)
    {
        for(int col = 0; col < n; ++col)
        {
            b[col + row * n] = (col * 2 + row + 3) % 256;
            printf("%3d ",b[col + row * n]);
        }
        printf("\n");
    }

    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    int loopNum = 10000;
	for(int i = 0; i < loopNum; i++)
	{
        matrix_multiplication_serial_1(a, b, c, m, n, l);
    }
    gettimeofday(&endTime, NULL);
	printf("matrix_multiplication_serial_1 use time: %d\n",
        (endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec));
        
    gettimeofday(&startTime, NULL);
    for(int i = 0; i < loopNum; i++)
	{
        matrix_multiplication_serial_2(a, b, c, m, n, l);
    }
    gettimeofday(&endTime, NULL);
    printf("matrix_multiplication_serial_2 use time: %d\n",
        (endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec));
    
    gettimeofday(&startTime, NULL);
    for(int i = 0; i < loopNum; i++)
	{
        matrix_multiplication_serial_3(a, b, c, m, n, l);
    }
    gettimeofday(&endTime, NULL);
    printf("matrix_multiplication_serial_3 use time: %d\n",
        (endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec));
    
    //result
    printf("result:\n");
    for(int row = 0; row < m; ++row)
    {
        for(int col = 0; col < n; ++col)
        {
            printf("%5d ",c[col + row * n]);
        }
        printf("\n");
    }
    return 0;
}
