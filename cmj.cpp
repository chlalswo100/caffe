#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int ***getActivation(int rows,int cols,int stride,int fil_size,int ***output,int channel)
{   //sigmoid function

    int *** sigmoid;
    sigmoid=(int ***)malloc(channel*sizeof(int **));
    for(int i=0;i<channel;i++)
    {
        *(sigmoid+i)=(int **)malloc( rows*sizeof(int *));
        for(int j=0;j<rows;j++)
        {    
            *(*(sigmoid+i)+j)=(int *)malloc(cols*sizeof(int));
        }                 
    }

    for(int k=0;k<channel;k++)
        {
            for(int a=0;a<rows;a++)
            {
                for(int b=0;b<cols;b++)
                {  
                    sigmoid[k][a][b] = 1/(1+exp(-output[k][a][b]));
                }   
            } 
        }
    return sigmoid;

    for(int i=0;i<channel;i++)
    {
        for(int j=0;j<rows;j++)
        {
            free(*(*(sigmoid+i)+j));
        }
        free(*(sigmoid+i));
    }
    free(sigmoid);
}

int ***conv(int rows,int cols,int ***output,int stride,int fil_size,double fil_sum,double **fil,int ***im_p,int channel)
{   
    
    int c=0;
    //Mat image=imread();     image.rows  image.cols  image.channels()
    for(int k=0;k<channel;k++)
        {
            clock_t begin = clock();
        
            for(int a=0;a<rows;a++)
            {
                for(int b=0;b<cols;b++)
                {  
                    for(int i=0;i<fil_size;i++)
                    {        
                        for(int j=0;j<fil_size;j++)
                        {
                            if(fil_sum>1)
                            output[k][a][b] += im_p[k][i+a*c][j+b*c]*fil[i][j]/fil_sum;
                            else
                            output[k][a][b] += im_p[k][i+a*c][j+b*c]*fil[i][j];
                        }
                    }
                c=stride;
                }   
            } 
        clock_t end = clock();
        double elapsed_secs = double(end-begin);
        printf("\nTime Check : %f\n",elapsed_secs);
        }
 return output;  
}

int ***Max_pooling(int rows,int cols,int stride,int fil_size,double **fil,int ***im_p,int channel)
{

    int *** max;
    max=(int ***)malloc(channel*sizeof(int **));
    for(int i=0;i<channel;i++)
    {
        *(max+i)=(int **)malloc(rows*sizeof(int *));
        for(int j=0;j<rows;j++)
        {    
            *(*(max+i)+j)=(int *)malloc(cols*sizeof(int));
        }
    }
    for(int k=0;k<channel;k++)
    {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {              
                max[k][i][j]=0;
            }  
        }  
    }
    for(int i=0;i<fil_size;i++)
    {
        for(int j=0;j<fil_size;j++)
        {              
            fil[i][j]=1;
        }  
    }  
    
    int c=0;
    for(int k=0;k<channel;k++)
        {
            clock_t begin = clock();
        
            for(int a=0;a<rows;a++)
            {
                for(int b=0;b<cols;b++)
                {  
                    for(int i=0;i<fil_size;i++)
                    {        
                        for(int j=0;j<fil_size;j++)
                        {
                            
                            if(max[k][a][b]<im_p[k][i+a*c][j+b*c]*fil[i][j])
                                max[k][a][b] = im_p[k][i+a*c][j+b*c]*fil[i][j];
                            else          
                                max[k][a][b] = max[k][a][b];
                        }
                    }
                c=stride;
                }   
            } 
            clock_t end = clock();  
            double elapsed_secs = double(end-begin);
            printf("\nTime Check : %f\n",elapsed_secs);
        }
        

    return max; 
    for(int i=0;i<channel;i++)
    {
        for(int j=0;j<rows;j++)
        {
            free(*(*(max+i)+j));
        }
        free(*(max+i));
    }
    free(max);
}

void conv2(Mat src,Mat image, int kernel_size,int channel,int out_size_rows,int out_size_cols,int ***output)
{
    Mat dst, kernel;
    //kernel = Mat::ones(kernel_size,kernel_size,CV_32F)/(double)(kernel_size*kernel_size);
    kernel = (Mat_<double>(3,3) << -1,-1,-1,-1,8,-1,-1,-1,-1);
    filter2D(src,dst,-1,kernel,Point(-1,-1),0,BORDER_DEFAULT);
    namedWindow("original",CV_WINDOW_AUTOSIZE);
    imshow("original",src);

    namedWindow("filter2D conv",CV_WINDOW_AUTOSIZE);
    imshow("filter2D conv",dst);
    
    Mat outImage2(out_size_rows,out_size_cols, CV_8UC3);

    for(int k=0;k<channel;k++)
    {
        for(int i=0;i<50;i++)
        {
            for(int j=0;j<50;j++)
            {  
                outImage2.at<cv::Vec3b>(i,j)[k]=dst.at<cv::Vec3b>(i,j)[k]-output[k][i][j];
                printf("입력받은 이미지 : %d,%d \n 컨벌루션한 이미지 : %d,%d \n 비교한 값 : %d\n\n\n\n",src.at<cv::Vec3b>(i,j)[k],image.at<cv::Vec3b>(i,j)[k],dst.at<cv::Vec3b>(i,j)[k],output[k][i][j],outImage2.at<cv::Vec3b>(i,j)[k]);
            }       
        }  
    }

    namedWindow("real compare",CV_WINDOW_AUTOSIZE);
    imshow("real compare",outImage2);
    
    
}

int main(){
    // 변수 선언
    int fil_size,pad_size,stride,q;
    int i,j,k,num,channel,mod;
    int ***im;
    int ***im_p;
    double **fil;
    double fil_sum;
    int out_size_rows,out_size_cols;
    int ***output;
    int **output2;
    // 필터 패딩 스트라이드 값 입력
    printf("filter size값을 입력하세요:");
    scanf("%d",&fil_size);
    printf("\n");

    printf("channel값을 입력하세요:");
    scanf("%d",&channel);
    printf("\n");

    printf("padding값을 입력하세요:");
    scanf("%d",&pad_size);
    printf("\n");

    printf("stride값을 입력하세요:");
    scanf("%d",&stride);
    printf("\n");

    printf("mod를 입력하세요(1은 max pooling 그외의 값 입력시 convolution):");
    scanf("%d",&mod);
    printf("\n");

    // 여기부터 이미지 읽어서 컨벌루션해주기

    Mat image;
    
    image = imread("test.jpg",IMREAD_COLOR);
    if(image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }
    
    for(int y=0;y<image.rows;y++)
    {
        for(int x=0;x<image.cols;x++)
        {
            cv::Vec3b pixel= image.at<cv::Vec3b>(y,x);
            image.at<cv::Vec3b>(y,x)[2] = image.at<cv::Vec3b>(y,x)[2];   //b
            image.at<cv::Vec3b>(y,x)[1] = image.at<cv::Vec3b>(y,x)[1];   //g
            image.at<cv::Vec3b>(y,x)[0] = image.at<cv::Vec3b>(y,x)[0];   //r
        }
    }


    // 출력 이미지 사이즈
    out_size_rows = ((image.rows-fil_size+(2*pad_size))/stride) + 1;   
    out_size_cols = ((image.cols-fil_size+(2*pad_size))/stride) + 1;   
    
    // 입력 이미지 할당
    im=(int ***)malloc(channel*sizeof(int **));
    for(i=0;i<channel;i++)
    {
        *(im+i)=(int **)malloc(image.rows*sizeof(int *));
        for(j=0;j<image.rows;j++)
        {
            *(*(im+i)+j)=(int *)malloc(image.cols*sizeof(int));
        }
    }

    // 패딩을 위한 할당
    im_p=(int ***)malloc(channel*sizeof(int **));
    for(i=0;i<channel;i++)
    {
        *(im_p+i)=(int **)malloc((image.rows+(2*pad_size))*sizeof(int *));
        for(j=0;j<image.rows+(2*pad_size);j++)
        {
            *(*(im_p+i)+j)=(int *)malloc((image.cols+(2*pad_size))*sizeof(int));
        }
    }

    // 필터 할당
    fil=(double **)malloc(fil_size*sizeof(double *));
    for(i=0;i<fil_size;i++)
    {
        *(fil+i)=(double *)malloc(fil_size*sizeof(double));
    }
   
    output2=(int **)malloc(out_size_rows*sizeof(int *));
    for(i=0;i<out_size_rows;i++)
    {
        *(output2+i)=(int *)malloc(out_size_cols*sizeof(int));
    }

    
    // 출력 할당
    output=(int ***)malloc(channel*sizeof(int **));
    //output2=(int ***)malloc(channel*sizeof(int **));
    for(i=0;i<channel;i++)
    {
        *(output+i)=(int **)malloc(out_size_rows*sizeof(int *));
      //  *(output2+i)=(int **)malloc(out_size_rows*sizeof(int *));
        for(j=0;j<out_size_rows;j++)
        {    
            *(*(output+i)+j)=(int *)malloc(out_size_cols*sizeof(int));
        //    *(*(output2+i)+j)=(int *)malloc(out_size_cols*sizeof(int));
        }
    }

    // 출력 배열 초기화                          
    for(k=0;k<channel;k++)
    {
        for(i=0;i<out_size_rows;i++)
        {
            for(j=0;j<out_size_cols;j++)
            {  
                output[k][i][j]=0;  
            }  
        }  
    }

    // 입력 이미지에 zero-padding                                 
    for(k=0;k<channel;k++)
    {
        for(i=0;i<image.rows+(2*pad_size);i++)
        {
            for(j=0;j<image.cols+(2*pad_size);j++)
            {
                im_p[k][i][j]=0;
            }
        }   
    }  
    // 필터 입력해주기
   
    printf("필터 값을 입력해주세요 :\n");

    for(i=0;i<fil_size;i++){
        for(j=0;j<fil_size;j++){
            scanf("%d",&num);        
            fil[i][j]=num;
        }   
    }	
    for(i=0;i<fil_size;i++){
        for(j=0;j<fil_size;j++){
            fil_sum += fil[i][j];
            // if(fil_sum==0)
            //     fil_sum=1;
            // else
            //     fil_sum=fil_sum;
        }   
    }	

    // 이미지값 쓰기
     for(k=0;k<channel;k++)
    {
        for(i=0;i<image.rows;i++)
        {
            for(j=0;j<image.cols;j++)
            {   // k=0 : r / k=1 : g / k=2 : b
                im[k][i][j]=image.at<cv::Vec3b>(i,j)[k];  
            }  
        }  
    }

    // 패딩해준거에다가 읽은 이미지 덧씌우기
    for(k=0;k<channel;k++)
    {
        for(i=0;i<image.rows;i++)
        {
            for(j=0;j<image.cols;j++)
            {
                im_p[k][i+pad_size][j+pad_size] = im[k][i][j];
            }   
        }	    
    }

    if(mod==1){
    output=Max_pooling(out_size_rows,out_size_cols,stride,fil_size,fil,im_p,channel);}
    else{
    output=conv(out_size_rows,out_size_cols,output,stride,fil_size,fil_sum,fil,im_p,channel);}
    

    for(int k=0;k<channel;k++) 
    {
        for(int a=0;a<out_size_rows;a++)
        {
            for(int b=0;b<out_size_cols;b++)
            {              
                if(output[k][a][b]<0){
                    output[k][a][b]=0;}
                else if(output[k][a][b]>255){
                    output[k][a][b]=255;}
                else{
                    output[k][a][b]=output[k][a][b];}
            }  
        }  
    }
    
    Mat src;
    //Load an image
    src = imread("test.jpg");
    if(!src.data){return -1;}

    conv2(src,image,3,channel,out_size_rows,out_size_cols,output);
    

    Mat outImage(out_size_rows,out_size_cols, CV_8UC3);

    for(k=0;k<channel;k++)
    {
        for(i=0;i<out_size_rows;i++)
        {
            for(j=0;j<out_size_cols;j++)
            {  
                //outImage.at<cv::Vec3b>(i,j)[k]=activation(output[k][i][j]);  
                outImage.at<cv::Vec3b>(i,j)[k]=output[k][i][j];  
            }  
        }  
    }
    
    
    namedWindow("myConv",WINDOW_AUTOSIZE);
    imshow("myConv",outImage);

    if(mod==1){
        //imwrite("Max_jong.jpg",outImage);}
        imwrite("Max_DangDangE.jpg",outImage);}
    else{
        //imwrite("Conv_jong.jpg",outImage);}
        imwrite("Conv_DangDangE.jpg",outImage);}
    waitKey(0);
    return 0;
    // 할당해준만큼 free해주기
    for(i=0;i<channel;i++)
    {
        for(j=0;j<image.rows+(2*pad_size);j++)
        {
            free(*(*(im_p+i)+j));
        }
        free(*(im_p+i));
    }
    free(im_p);

    for(i=0;i<fil_size;i++)
    {
        free(*(fil+i));
    }

    for(i=0;i<channel;i++)
    {
        for(j=0;j<image.rows;j++)
        {
            free(*(*(im+i)+j));
        }
        free(*(im+i));
    }
    free(im);
    for(i=0;i<channel;i++)
    {
        for(j=0;j<out_size_rows;j++)
        {
            free(*(*(output+i)+j));
        }
        free(*(output+i));
    }
    free(output);
        
}