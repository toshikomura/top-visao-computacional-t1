#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

void readme();

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_1 = imread( argv[ 1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[ 2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  Size s1 = img_1.size();
  Size s2 = img_2.size();

  int row1 = s1.height;
  int colum1 = s1.width;
  int row2 = s2.height;
  int colum2 = s2.width;

  printf ( "First image size %d channels %d\n", row1*colum1, img_1.channels());
  printf ( "Second image size %d channels %d\n", row2*colum2, img_2.channels());

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;

  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow( "Keypoints 1", img_keypoints_1 );
  imshow( "Keypoints 2", img_keypoints_2 );

  waitKey( 0);

  return 0;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./detection <img1> <img2>" << std::endl; }
