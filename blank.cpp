#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include "/usr/include/opencv2/features2d/features2d.hpp"
#include "/usr/include/opencv2/core/core.hpp"

int main(int argc, char *argv[])
{
  //IplImage* img = 0;
  cv::Mat img_keypoints_1;
  int height, width, step, channels;
  uchar *data;
  int i,j,k;

  // load an image
  //img = cvLoadImage( "20130114_160137-low-contrast.jpg");
  cv::Mat img = imread( argv[1],CV_LOAD_IMAGE_GRAYSCALE);

  if( !img){
    printf( "Could not load image file: %s\n", argv[ 1]);
    exit( 0);
  }

  // get the image data
  height    = img->height;
  width     = img->width;
  step      = img->widthStep;
  channels  = img->nChannels;
  data      = (uchar *)img->imageData;
  printf("Processing a %dx%d image with %d channels with step %d\n", height, width, channels, step);

  // create a window
  cvNamedWindow( "example2", CV_WINDOW_AUTOSIZE);
  //cvMoveWindow( "example2", 100, 100);    // to move the window to position 100 100.

  // invert the image
  for( i =0; i < height; i++)
     for( j =0; j < width; j++)
        for( k =0; k < channels; k++) {  //loop to read for each channel
           data[ i * step + j * channels + k] = 255 - data[ i * step + j * channels + k];    //inverting the image
        }

  cv::SurfFeatureDetector detector( 400, 3, 2, false);

  std::vector<cv::KeyPoint> keypoints_1;

  detector.detect( img, keypoints_1 );

  drawKeypoints( img, keypoints_1, img_keypoints_1);

  printf ( "keypoints %d\n", keypoints_1.lenght);

  cv::imshow("Keypoints1.jpg", img_keypoints_1 );

  // show the image
  cvSaveImage( "inversao.jpg", img );

  // wait for a key
  cvWaitKey( 0);

  // release the image
  cvReleaseImage( &img );
  cvDestroyWindow( "example2");
  return 0;
}
