//
//  ViewController.h
//  My_MonoVO_iOS3
//
//  Created by Peter on 12/4/16.
//  Copyright © 2016 Carnegie Mellon University. All rights reserved.
//

#import <UIKit/UIKit.h>
//#import <opencv2/highgui/ios.h>
#import <opencv2/videoio/cap_ios.h>

@interface ViewController : UIViewController<CvVideoCameraDelegate>
{
    CvVideoCamera *videoCamera; // OpenCV class for accessing the camera
}
// Declare internal property of videoCamera
@property (nonatomic, retain) CvVideoCamera *videoCamera;

@end

