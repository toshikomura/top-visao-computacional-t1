#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

void readme();

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 2 )
  { readme(); return -1; }

  Mat imput_img = imread( argv[ 1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat warp_rotate_dst, rot_mat;
  Mat equalize_img;
  Mat img_matches;

  Size s = imput_img.size();

  int row = s.height;
  int colum = s.width;

  Mat mask;
  equalizeHist( imput_img, mask);

  int i, j;

  for ( i = 0; i < row * colum; i++)
      mask.data[i] = 0;

  for ( i = (row * colum)/2 - (row * colum)/200; i < (row *colum )/2 + (row * colum)/150; i = i + 2)
    mask.data[i] = 1;

  printf ( "Imput image size %d %d %d channels %d\n", row, colum, row *colum, imput_img.channels());

  /// Apply Histogram Equalization
  equalizeHist( imput_img, equalize_img);

  /// Compute a rotation matrix with respect to the center of the image
  Point center = Point( row/2, colum/2 );
  double angle = 45.0;
  double scale = 1.0;

  /// Get the rotation matrix with the specifications above
  rot_mat = getRotationMatrix2D( center, angle, scale );

  /// Rotate image
  warpAffine( equalize_img, warp_rotate_dst, rot_mat, imput_img.size() );

  //-- Step 1: Detect the keypoints using SURF Detector
  printf ( "Prepare Surf\n");
  SurfFeatureDetector detector( 400, 3, 2, false);

  std::vector<KeyPoint> keypoints_imput, keypoints_equalize, keypoints_rotate;

  printf ( "Detect points\n");
  detector.detect( imput_img, keypoints_imput, mask);
  detector.detect( warp_rotate_dst, keypoints_equalize, mask);
  detector.detect( equalize_img, keypoints_rotate, mask);

  // computing descriptors
  printf ( "Computing descriptors\n");
  SurfDescriptorExtractor extractor;
  Mat descriptors1, descriptors2, descriptors3;
  extractor.compute( imput_img, keypoints_imput, descriptors1);
  extractor.compute( equalize_img, keypoints_equalize, descriptors2);
  extractor.compute( warp_rotate_dst, keypoints_rotate, descriptors3);

  // matching descriptors
  printf ( "Matching descriptors\n");
  BruteForceMatcher<L2<float> > matcher;
  vector<DMatch> matches;
  matcher.match( descriptors1, descriptors2, matches);

  // drawing the results
  drawMatches( imput_img, keypoints_imput, equalize_img, keypoints_equalize, matches, img_matches);
  imwrite( "matches1_equalize.jpg", img_matches);

  matcher.match( descriptors1, descriptors3, matches);
  drawMatches( imput_img, keypoints_imput, warp_rotate_dst, keypoints_rotate, matches, img_matches);
  imwrite( "matches1_rotate.jpg", img_matches);

  //-- Draw keypoints
  Mat keypoints_imput_img, keypoints_equalize_img, keypoints_rotate_img;

  drawKeypoints( imput_img, keypoints_imput, keypoints_imput_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( equalize_img, keypoints_equalize, keypoints_equalize_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( warp_rotate_dst, keypoints_rotate, keypoints_rotate_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imwrite( "keypoints_imput_img.jpg", keypoints_imput_img);
  imwrite( "keypoints_equalize_img.jpg", keypoints_equalize_img);
  imwrite( "keypoints_rotate_img.jpg", keypoints_rotate_img);

  waitKey( 0);

  return 0;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./detection <img1> <img2>" << std::endl; }
