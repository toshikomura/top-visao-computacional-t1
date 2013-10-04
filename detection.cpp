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

  Mat img_1 = imread( argv[ 1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat warp_rotate_dst, rot_mat;
  Mat equalize_img;

  Size s1 = img_1.size();

  int row1 = s1.height;
  int colum1 = s1.width;

  Mat mask;
  equalizeHist( img_1, mask);

  int i, j;

  for ( i = 0; i < row1*colum1; i++)
      mask.data[i] = 0;

  for ( i = (row1*colum1)/3 + 5800; i < (row1*colum1)/2 + 20; i++)
    mask.data[i] = 10;

  printf ( "First image size %d %d %d channels %d\n", row1, colum1, row1*colum1, img_1.channels());

  /// Apply Histogram Equalization
  equalizeHist( img_1, equalize_img);

  Size s2 = equalize_img.size();

  int row2 = s2.height;
  int colum2 = s2.width;

  /// Compute a rotation matrix with respect to the center of the image
  Point center = Point( row2/2, colum2/2 );
  double angle = 45.0;
  double scale = 1.0;

  /// Get the rotation matrix with the specifications above
  rot_mat = getRotationMatrix2D( center, angle, scale );

  /// Rotate the warped image
  warpAffine( equalize_img, warp_rotate_dst, rot_mat, img_1.size() );

  Size s3 = warp_rotate_dst.size();

  int row3 = s3.height;
  int colum3 = s3.width;

  printf ( "row %d colum %d\n", row3, colum3);

  //-- Step 1: Detect the keypoints using SURF Detector

  SurfFeatureDetector detector( 400, 3, 2, false);

  std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_3;

  detector.detect( img_1, keypoints_1, mask);
  detector.detect( warp_rotate_dst, keypoints_2, mask);
  detector.detect( equalize_img, keypoints_3, mask);

  // computing descriptors
  SurfDescriptorExtractor extractor;
  Mat descriptors1, descriptors2, descriptors3;
  extractor.compute( img_1, keypoints_1, descriptors1);
  extractor.compute( equalize_img, keypoints_2, descriptors2);
  extractor.compute( warp_rotate_dst, keypoints_3, descriptors3);

  // matching descriptors
  BruteForceMatcher<L2<float> > matcher;
  vector<DMatch> matches;
  matcher.match( descriptors1, descriptors2, matches);

  // drawing the results
  namedWindow( "matches", 1);
  Mat img_matches;
  drawMatches( img_1, keypoints_1, equalize_img, keypoints_2, matches, img_matches);
  imshow( "matches1_equalize", img_matches);

  matcher.match( descriptors1, descriptors3, matches);
  drawMatches( img_1, keypoints_1, warp_rotate_dst, keypoints_3, matches, img_matches);
  imshow( "matches1_rotate", img_matches);

  //-- Draw keypoints
  Mat img_keypoints_1, img_keypoints_2, img_keypoints_3;

  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( warp_rotate_dst, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( equalize_img, keypoints_3, img_keypoints_3, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow( "Keypoints 1", img_keypoints_1 );
  imshow( "2", img_keypoints_2);
  imshow( "3", img_keypoints_3);

  waitKey( 0);

  return 0;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./detection <img1> <img2>" << std::endl; }
