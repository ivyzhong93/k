//
//  ViewController.m
//  My_MonoVO_iOS
//
//  Created by Peter on 11/26/16.
//  Copyright © 2016 Carnegie Mellon University. All rights reserved.
//

#import "ViewController.h"

#ifdef __cplusplus
#include <opencv2/opencv.hpp> // Includes the opencv library
#include <stdlib.h> // Include the standard library
//#include "armadillo" // Includes the armadillo library
#include "vo_features.h"

#endif

using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 500

@interface ViewController ()
{
    UIImageView *cameraView;
    UIImageView *trajectoryView;
    IBOutlet UIButton *resetButton;

    int numFrame;
    int frameN;
    NSTimer *timer;
    Mat prevImage;
    vector<Point2f> prevFeatures;
    double focal;
    cv::Point2d pp;
    Mat R_f, t_f; //the cumulated rotation and tranlation vectors
    Mat traj;
}
@end


@implementation ViewController

// Important as when you when you override a property of a superclass, you must explicitly synthesize it
@synthesize videoCamera;

//improvement: camera calibration, undistort, IMU


- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    // Set intrinsics
    focal = 1187;
//    pp = cv::Point2d(240, 320);
//    pp = cv::Point2d(645.92383, 359.28473);
//    pp = cv::Point2d(645.92383, 359.28473);
    pp = cv::Point2d(320, 240);

    
    // set up output window, text and initialize trajectory matrix
    int viewWidth = self.view.frame.size.width;
    int viewHeight = self.view.frame.size.height;
//    cameraView = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, viewHeight * 360 / 1280, viewHeight/2)];
//    cameraView = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, viewWidth, viewWidth * 720 / 1280)];
    cameraView = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, viewWidth, viewWidth * 480 / 640)];

    cameraView.contentMode = UIViewContentModeScaleAspectFit; // Set contentMode(optional)
    [self.view addSubview:cameraView]; // Important: add liveView_ as a subview
    trajectoryView = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, viewHeight/2, viewWidth, viewHeight/2)];
    trajectoryView.contentMode = UIViewContentModeScaleAspectFit; // Set contentMode(optional)
    [self.view addSubview:trajectoryView]; // Important: add liveView_ as a subview

    // set initial videoView and trajectoryView images
    traj = Mat::zeros(600, 600, CV_8UC3);
    cv::flip(traj, traj, 0);
    trajectoryView.image = [self UIImageFromCVMat:traj];

    // Initialize the video camera
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:cameraView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeRight;//AVCaptureVideoOrientationPortrait;//put the camera view into a matrix according to your view described as the attribute.
    self.videoCamera.defaultFPS = 30; // Set the frame rate
    self.videoCamera.grayscaleMode = NO; // Get grayscale
    self.videoCamera.rotateVideo = YES; // Rotate video so everything looks correct ???
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;//AVCaptureSessionPreset1280x720;

    numFrame = 0;
    frameN = 0;
    t_f = Mat::zeros(3, 1, CV_64FC1);
    R_f = Mat::eye(3, 3, CV_64FC1);
    
    [self.view addSubview:resetButton]; // Important: add the button as a subview

    // Finally show the output
    [videoCamera start];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (IBAction)buttonWasPressed:(id)sender{
    traj = Mat::zeros(600, 600, CV_8UC3);
    t_f = Mat::zeros(3, 1, CV_64FC1);
    R_f = Mat::eye(3, 3, CV_64FC1);
    numFrame = 0;
}

- (void) processImage:(cv:: Mat &)image
{
    if (frameN < 30) {
        frameN++;
        return;
    }
    
//    if (numFrame == 0) {
//        cout << image.rows << endl;
//        cout << image.cols << endl;
//    }

//    transpose(image, image);
//    cv::Mat currImage;
//    cvtColor(image, currImage, COLOR_BGR2GRAY);

    Mat currImage;
    if(image.channels() == 4)
        cvtColor(image, currImage, CV_RGBA2GRAY); // Convert to grayscale
    else
        currImage = image;

    if (numFrame == 0) {
        cout << "Redection on new image..." << endl;
        prevImage = currImage.clone();
        featureDetection(prevImage, prevFeatures);
        if(prevFeatures.size() > MIN_NUM_FEAT)
            numFrame++;
        return;
    }

//    cout << "feature number: " << prevFeatures.size() << endl;

    // compute R,t by tracking features
    vector<Point2f> currFeatures;
    vector<uchar> status;
    Mat R, t, mask;
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
    Mat E = findEssentialMat(prevFeatures, currFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
    if (E.cols != 3 || E.rows != 3) {
        cout << "E = " << E << endl;
        R = Mat::eye(3, 3, CV_64FC1);
        t = Mat::zeros(3, 1, CV_64FC1);
    }
    else
        recoverPose(E, prevFeatures, currFeatures, R, t, focal, pp, mask);

//    if (t.at<double>(2) > 0)
//        t = -t;

    cout << "R = " << R << endl;
    cout << "t = " << t << endl;

    double scale = 2;
//    if ((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
//    if ((abs(t.at<double>(2)) > abs(t.at<double>(0))) && (abs(t.at<double>(2)) > abs(t.at<double>(1)))) { // is it good ???
        Mat R_i; cv::transpose(R, R_i);
        t_f = t_f + scale * (R_f * -R_i * t);//t_f = t_f + scale*(R_f*t);
        R_f = R_i * R_f;//R_f = R*R_f;
//    }
    
    cout << "R_f = " << R_f << endl;
    cout << "t_f = " << t_f << endl << endl;

    // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
    if (prevFeatures.size() < MIN_NUM_FEAT)	{
        cout << "trigerring redection..." << endl;
        featureDetection(currImage, currFeatures);
        if(prevFeatures.size() < MIN_NUM_FEAT) {
            numFrame = 0;
        }
//        featureDetection(prevImage, prevFeatures);
//        if(prevFeatures.size() < MIN_NUM_FEAT) {
//            numFrame = 0;
//        }
//        else {
//            featureTracking(prevImage,currImage,prevFeatures,currFeatures, status); // why use prevImage rather than currImage here ?????
//        }
    }
    
    // update prevImage and prevFeatures for next loop
    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    // draw on trajectoryView and update videoView and trajectoryView
    drawing(traj, t_f);
    Mat traj_flip;
    cv::flip(traj, traj_flip, 0);
    dispatch_sync(dispatch_get_main_queue(), ^{
        trajectoryView.image = [self UIImageFromCVMat:traj_flip];
    });
    
    vector<KeyPoint> currFeatures_k;
    KeyPoint::convert(currFeatures, currFeatures_k);
    cv::drawKeypoints(currImage, currFeatures_k, image, Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
}

void drawing(cv::Mat& traj, cv::Mat t_f) {
    // drawing trajectory
    int x = int(t_f.at<double>(0)) + 300;
    int z = int(t_f.at<double>(2)) + 100;
//    cout << " (x = " << x << ", z = " << z << ")" << endl;
    cv::circle(traj, cv::Point(x, z) ,1, Scalar(0, 255, 0), 2);// green is our trajetory
    
    
    // put text on trajectory view
//    char text[100];
//    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
//    cv::putText(traj, text, cv::Point(10, 50), FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(255), 1, 8);
}


bool drawGroundTruth(cv::Mat& traj, int num) {
    NSString *filePath = [[NSBundle mainBundle] pathForResource:@"00" ofType:@"txt"];
    string path = [filePath UTF8String];
    ifstream myfile(path);
    
    int i = 0;
    double x = 0, y = 0, z = 0;
    string line;
    if (myfile.is_open()) {
        while (getline(myfile,line) && i < num) { // traverse poses and draw ground truth trajetory
            std::istringstream in(line);
            for (int j = 0; j < 12; j++)  {
                in >> z;
                if (j==3)
                    x = z;
                if (j==7)
                    y = z;
            }
            cv::circle(traj, cv::Point(x + 300, z + 100) ,1, cv::Scalar(0, 0, 255), 2);// use blue for ground truth trajetory
            i++;
        }
        myfile.close();
        return true;
    }
    else {
        cout << "Unable to open file";
        return false;
    }
}

double getAbsoluteScale(int frame_id) {
    
    string line;
    int i = 0;
    
    NSString *filePath = [[NSBundle mainBundle] pathForResource:@"00" ofType:@"txt"];
    string path = [filePath UTF8String];
    ifstream myfile(path);
    
    double x =0, y=0, z = 0;
    double x_prev = 0, y_prev = 0, z_prev = 0;
    if (myfile.is_open())
    {
        while (( getline (myfile,line) ) && (i<=frame_id))
        {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j=0; j<12; j++)  {
                in >> z ;
                if (j==7) y=z;
                if (j==3)  x=z;
            }
            
            i++;
        }
        myfile.close();
    }
    
    else {
        cout << "Unable to open file";
        return 0;
    }
    
    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
}


//[trajectoryView performSelectorOnMainThread:@selector(setImage:) withObject: [self UIImageFromCVMat:traj] waitUntilDone:YES];
//[videoView performSelectorOnMainThread:@selector(setImage:) withObject: [self UIImageFromCVMat:img_2_c] waitUntilDone:YES];
//[trajectoryView setNeedsDisplay];
//[videoView setNeedsDisplay];
//- (void)setImage {}


// Member functions for converting from UIImage to cvMat
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}
// Member functions for converting from cvMat to UIImage
-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

@end
