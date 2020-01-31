#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <unistd.h>
#include <string>

//using namespace cv;
using namespace std;

#ifdef WIN32
#define OS_SEP '\\'
#else
#define OS_SEP '/'
#endif

#define PI 3.14159265359


cv::Mat cutOutUnnecessaryParts(cv::Mat img){
    /// <summary>draws a contour into the areas not captured by the camera (left and right bottom corner) in order to
    /// get rid of edges detected by the canny algorithm</summary>
    /// <param name="img">Image in which contour is to be drawn</param>
    /// <returns>adapted Image</returns>
    int height = int(img.rows);
    int width = int(img.cols);
    vector<cv::Point> trimaskRightCorner;
    trimaskRightCorner.push_back(cv::Point(width/2 +50, height));
    trimaskRightCorner.push_back(cv::Point(width, 0));
    trimaskRightCorner.push_back(cv::Point(width, height));


    vector<cv::Point> trimaskLeftCorner;
    trimaskLeftCorner.push_back(cv::Point(width/2 -50, height));
    trimaskLeftCorner.push_back(cv::Point(0, height/2 -30));
    trimaskLeftCorner.push_back(cv::Point(0, height));

    vector<vector< cv::Point> > contours;
    contours.push_back(trimaskRightCorner);
    contours.push_back(trimaskLeftCorner);
    cv::Mat result = img.clone();
    drawContours(result, contours, -1, (0,255,0), cv::FILLED);
    return result;
}


cv::Mat canny(cv::Mat img, int lower= 50, int upper = 120){
    /// <summary>applies the canny edge algorithm to the given image</summary>
    /// <param name="img">image in which edges are to be detected</param>
    /// <param name="lower">lower threshold of canny edge detection</param>
    /// <param name="upper">upper threshold of canny edge detection</param>
    /// <returns>Image containing detected edges</returns>
    cv::Mat output;
    Canny(img, output, lower, upper);
    return output;
}

// DO EXPERIMENTAION!!
cv::Mat smoothe(cv::Mat image, int kernelSize=15){
    /// <summary>applies gaussian blur to given image</summary>
    /// <param name="img">image to which gaussian blur is to be applied</param>
    /// <param name="kernelSize">lower threshold of canny edge detection</param>
    /// <returns>blurred image</returns>
    cv::Mat smoothedImage;
    cv::GaussianBlur(image, smoothedImage, cv::Size(kernelSize, kernelSize), 0);
    return  smoothedImage;
}


vector<cv::Vec4i> houghLines(cv::Mat img, int rho=1, int theta=PI/180, int threshold=30, int minLineLength=20, int maxLineGap=10) {
    /// <summary>applies the hough line detection to image</summary>
    /// <param name="img">image in which lines are to be detected</param>
    ///for other params see. https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlines
    /// <returns>Image containing detected lines</returns>
    vector<cv::Vec4i> lines;
    HoughLinesP(img, lines, rho, PI/180, threshold, minLineLength, maxLineGap);
    return lines;
}


bool findIntersection(cv::Point &interception, cv::Point o1, cv::Point d1, cv::Point o2, cv::Point d2){
    /// <summary>find the intersection of two lines or returns false if no intersection
    /// (infinite lines intersection)</summary>
    /// <param name="interception">variable in which interception gets saved</param>
    /// lines are defined by (o1,d1) and (o2,d2)
    /// <returns>true if lines intercept else false/returns>
    cv::Point2f x = o2 - o1;
    cv::Point2f n1 = d1 - o1;
    cv::Point2f n2 = d2 - o2;

    float cross = n1.x*n2.y - n1.y*n2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * n2.y - x.y * n2.x)/cross;;
    interception = cv::Point2f(o1) + n1 * t1;

    return true;
}


int angle(cv::Point p1, cv::Point p2){
    /// <summary>calculates angle of line represented by (p1,p2)</summary>
    float angle = atan2(p1.y - p2.y, p1.x - p2.x);
    angle = angle*180/CV_PI;
    int ang = int(angle);
    return angle;
}


float euclideanDist(cv::Point p, cv::Point q) {
    /// <summary>calculates euclidean distance between two points p and q</summary>
    cv::Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}


float getDistance4p(cv::Point &r1, cv::Point &r2, cv::Point o1, cv::Point d1, cv::Point o2, cv::Point d2){
    /// <summary>calculates smallest distance between four points
    /// (smallest distance between points of two lines) </summary>
    /// lines are represenetd by (o1,d1) and (o2,d2)
    /// <param name="r1">return value for Point of line 1 with smallest distance</param>
    /// <param name="r2">return value for Point of line 2 with smallest distance</param>
    /// <returns>minimum distance of points between two lines/returns>
    float dist1 = euclideanDist(o1, o2);
    float dist2 = euclideanDist(o1, d2);
    float dist3 = euclideanDist(d1, o2);
    float dist4 = euclideanDist(d1, d2);

    // dont use min -> need to know which distance is being returned -
    //return min({dist1, dist2, dist3, dist4});
    if (min({dist1, dist2, dist3, dist4}) == dist1){
        r1 = o1;
        r2 = o2;
        return dist1;
    }
    else if (min({dist1, dist2, dist3, dist4}) == dist2){
        r1 = o1;
        r2 = d2;
        return dist2;
    }
    else if (min({dist1, dist2, dist3, dist4}) == dist3){
        r1 = d1;
        r2 = o2;
        return dist3;
    }
    else {
        r1 = d1;
        r2 = d2;
        return dist4;
    }
}


cv::Vec4i mergeLines(cv::Point l1o, cv::Point l1d, cv::Point l2o, cv::Point l2d){
    /// <summary>merges two lines</summary>
    cv::Point newO, newD;
    //coordinate system start at top
    //make dependent on curve as well    steep curve can lead to wrong origin + destination points

    if (l1o.y >= l2o.y){ //what about steep curves??
        newO = cv::Point(l1o);
    }
    else {
        newO = cv::Point(l2o);
    }
    if (l1d.y <= l2d.y){
        newD = cv::Point(l1d);
    }
    else {
        newD = cv::Point(l2d);
    }

    return cv::Vec4i(newO.x, newO.y, newD.x, newD.y);
}


bool belongTogether(cv::Vec4i line1, cv::Vec4i line2){
    /// <summary>checks if two line segments belong together</summary>
    cv::Point l1o = cv::Point(line1[0], line1[1]);
    cv::Point l1d = cv::Point(line1[2], line1[3]);

    cv::Point l2o = cv::Point(line2[0], line2[1]);
    cv::Point l2d = cv::Point(line2[2], line2[3]);

    cv::Point intersection;
    cv::Point closest11, closest21;
    if (findIntersection(intersection, l1o, l1d, l2o, l2d)){
//        if (getDistance4p(closest11, closest21, l1o, l1d, l2o, l2d) <= 30 and (abs(angle(l1o, l1d) - angle(l2o, l2d)) <= 20)){
//            return true;
//        }
        if (euclideanDist(l1d, l2o) <=50 and (abs(angle(l1o, l1d) - angle(l2o, l2d)) <= 20)){
            if(abs(l1d.x - l2o.x) <=  20){
                return true;
            }
            //return true;
        }
//        if (getDistance2p(l1d, intersection) <= 30 and getDistance2p(intersection, l2o) <=30 and (abs(angle(l1o, l1d) - angle(l2o, l2d)) <= 20)){
//            return true;
//            if(abs(l1d.x - l2o.x) <=  30){
//                return true;
//            }
//        }
    }

    cv::Point closest1, closest2;
    //for parallel lines that represent same line but dont intercept
    //dist 20 too small otherwise other line frags missedM; problem with 30 is that parallel lines with too mich distance on x axis get recognised
    //40 for second approach above
    if (((getDistance4p(closest1, closest2, l1o, l1d, l2o, l2d)) <= 20) and
             (abs(angle(l1o, l1d) - angle(l2o, l2d)) <= 10)) {
        //return true;
        if(abs(closest1.x - closest2.x) <=  20){
            return true;
        }
    }

    return false;
}


vector<cv::Vec4i> sortLinePoints(vector<cv::Vec4i> lineSegments){
    /// <summary>sort line fragments into (PointOrigin, PointDestination) form</summary>
    vector<cv::Vec4i> lineSegmentsSortedPoints = vector<cv::Vec4i>(lineSegments);

    auto it = lineSegmentsSortedPoints.begin();
    int size = lineSegmentsSortedPoints.size();
    int l = 0;
    while (l < size) {
        cv::Vec4i line1 = *it;
        cv::Point l1o, l1d;
        //coordinate system origin at top
        //determine origin an destination points
        if (line1[1] > line1[3]){
            l1o = cv::Point(line1[0], line1[1]);
            l1d = cv::Point(line1[2], line1[3]);
        }
        else{
            l1o = cv::Point(line1[2], line1[3]);
            l1d = cv::Point(line1[0], line1[1]);
        }
        it = lineSegmentsSortedPoints.erase(it);
        lineSegmentsSortedPoints.push_back(cv::Vec4i(l1o.x, l1o.y, l1d.x, l1d.y));
        l++;
    }
    return lineSegmentsSortedPoints;
}


struct comparator { //comparator struct for ordering lines on y axis
    // expects first two values of line to be the origin point of line
    bool operator() (cv::Vec4i line1, cv::Vec4i line2) { return (line1[1] >= line2[1]);}
} comp;


struct comparator2 { //comparator struct for ordering lines on y axis
    // expects first two values of line to be the origin point of line
    bool operator() (cv::Vec4i line1, cv::Vec4i line2) { return (line1[3] >= line2[3]);}
} comp2;


struct Line { //represents a line composed by line segments
    std::vector<cv::Vec4i> segs; //line segments
    Line(cv::Vec4i line){ segs.push_back(line);}
    const cv::Vec4i &getEnd() const { *segs.rbegin(); }
};


vector<Line> groupLineSegments(vector<cv::Vec4i> lineSegments){
    /// <summary>groups line-segments into lines</summary>
    vector<Line> groupedLines;
    for (cv::Vec4i seg : lineSegments) {
        bool added = false;
        for (Line &line : groupedLines) {
            if (belongTogether(line.getEnd(), seg)) {
                line.segs.push_back(seg);
                added = true;
                break;
                // break to next *segment*, a segment can only be added to one line.
            }
        }
        // reaching here means we didn't make attach the segment; start a new line.
        if (not added){
            groupedLines.push_back(Line(seg));
        }
    }
    return groupedLines;
}


cv::Mat drawLineFragments(cv::Mat img, vector<cv::Vec4i> lineSegments, cv::Scalar color = cv::Scalar(0,255,0)){
    /// <summary>draws lines into image</summary>
    /// <param name="img">image in which lines are to be drawn</param>
    /// <param name="lineFragments">line segments which are to be drawn</param>
    /// <param name="color">color of lines</param>
    /// <returns>Image containing drawn lines/returns>
    cv::Mat imgRgb(img.size(), CV_8UC3);
    cv::cvtColor(img, imgRgb, cv::COLOR_GRAY2RGB);

    for (int i = 0; i < lineSegments.size(); i++){
        cv::Vec4i l = lineSegments[i];
        line(imgRgb, cv::Point(l[0], l[1]), cv::Point(l[0], l[1]), cv::Scalar(255, 0, 0), 5);
        line(imgRgb, cv::Point(l[2], l[3]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 5);
        line(imgRgb, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), color, 2);
    }
    return imgRgb;
}


cv::Mat drawLanes(cv::Mat img, vector<cv::Vec4i> lineSegmentsRightLane, vector<cv::Vec4i> lineSegmentsLeftLane){
    /// <summary>draws lines into image</summary>
    /// <param name="img">image in which lines are to be drawn</param>
    /// <param name="lineFragments">line segments which are to be drawn</param>
    /// <param name="color">color of lines</param>
    /// <returns>Image containing drawn lines/returns>
    cv::Mat imgRgb(img.size(), CV_8UC3);
    cv::cvtColor(img, imgRgb, cv::COLOR_GRAY2RGB);

    for (int i = 0; i < lineSegmentsRightLane.size(); i++){
        cv::Vec4i l = lineSegmentsRightLane[i];
        line(imgRgb, cv::Point(l[0], l[1]), cv::Point(l[0], l[1]), cv::Scalar(0, 255, 0), 5);
        line(imgRgb, cv::Point(l[2], l[3]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 5);
        line(imgRgb, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 2);
    }

    for (int i = 0; i < lineSegmentsLeftLane.size(); i++){
        cv::Vec4i l = lineSegmentsLeftLane[i];
        line(imgRgb, cv::Point(l[0], l[1]), cv::Point(l[0], l[1]), cv::Scalar(0, 255, 0), 5);
        line(imgRgb, cv::Point(l[2], l[3]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 5);
        line(imgRgb, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2);
    }
    return imgRgb;
}


cv::Mat drawColouredLines(cv::Mat img, vector<Line> lines){
    int NumberOfLines = lines.size();

    //get a random color for each line
    cv::RNG rng(215546);
    std::vector<cv::Scalar> colors(NumberOfLines);
    for (int i = 0; i < NumberOfLines; i++){
        colors[i] = cv::Scalar(rng.uniform(30,255), rng.uniform(30, 255), rng.uniform(30, 255));;
    }

    cv::Mat imgRGB(img.size(), CV_8UC3);
    cvtColor(img, imgRGB, cv::COLOR_GRAY2RGB);


    for (int i = 0; i < lines.size(); i++){
        for (int j = 0; j < lines[i].segs.size(); j++) {
            cv::Vec4i &line = lines[i].segs[j];
            cv::line(imgRGB, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), colors[i], 3);
        }
    }
    return imgRGB;
}


void classifyLaneLines(vector<cv::Vec4i> &rightLaneLineSegments, vector<cv::Vec4i> &leftLaneLineSegments, vector<Line> lines){
    int approxCarPosX = 375;
    int approxCarPosY = 450;

    for (Line line: lines){
        if (line.segs.size() >= 2){
            cv::Vec4i startSeg = line.segs.front();

            if ((startSeg[1] > 350) /*y-axis*/ and (startSeg[0] > approxCarPosX + 20 and startSeg[0] < 465)){
                if(abs((angle(cv::Point(startSeg[0],startSeg[1]), cv::Point(startSeg[2], startSeg[3])) -90)) < 25){
                    if (rightLaneLineSegments.size() == 0) {
                        rightLaneLineSegments = line.segs;
                    }
                }
            }
            else if ((startSeg[1] > 350) /*y-axis*/ and (startSeg[0] < approxCarPosX-65 and startSeg[0] > approxCarPosX-180)){
                if(abs((angle(cv::Point(startSeg[0],startSeg[1]), cv::Point(startSeg[2], startSeg[3])) -90)) < 40){
                    if (leftLaneLineSegments.size() == 0){
                        leftLaneLineSegments = line.segs;
                    }
                }

            }
        }

    }
}


std::string getcwd_string( void ) {
    char buff[PATH_MAX];
    getcwd( buff, PATH_MAX );
    std::string cwd( buff );
    return cwd;
}


cv::Vec4i calculateNewEndLineSegment(cv::Vec4i oldEndLineSegment){
    float theta = atan2(oldEndLineSegment[1]-oldEndLineSegment[3], oldEndLineSegment[0]-oldEndLineSegment[2]);
    int endpt_x = int(oldEndLineSegment[2] - 1000*cos(theta));
    int endpt_y = int(oldEndLineSegment[3] - 1000*sin(theta));
    cv::Vec4i newEndLineSeg = cv::Vec4i(oldEndLineSegment[2] , oldEndLineSegment[3], endpt_x, endpt_y);
    return newEndLineSeg;
}



int main() {
    //get all images
    vector<cv::String> fn;
    string folderName = "malte-smashes-the-walls";
    string folderName2 = "footage_collection";
    string fpath = getcwd_string() + OS_SEP + folderName2 + OS_SEP + folderName +  OS_SEP + "*.png";

    cv::glob(fpath, fn, false);
    if (fn.empty()){
        cout << "Could not find images" << endl;
    }
    size_t count = fn.size();
    vector<cv::Vec4i> rightLaneLineSegments, leftLaneLineSegments;
    // iterate through all found images
    for (size_t i = 0; i < count; i++) {
        cout << fn[i] << endl;
        cv::Mat frame = imread(fn[i], cv::IMREAD_GRAYSCALE);

        cv::Mat smoothed = smoothe(frame);

        cv::Mat cannyimg  = canny(smoothed);


        cv::Mat dilated;
        cv::Mat eroded;

        cv::Mat kernel = getStructuringElement (cv::MORPH_ELLIPSE, cv::Point(9, 9));
        dilate(cannyimg, dilated, kernel);

        cv::Mat kernel1 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Point(12, 12));
        erode(dilated, eroded, kernel1);

        imshow("canny", cannyimg);
        imshow("eroded grouped", eroded);


        cv::Mat cut = cutOutUnnecessaryParts(cannyimg);

        vector<cv::Vec4i> lineSegments = houghLines(cut);

        vector<cv::Vec4i> lineSegmentsSortedPoints = sortLinePoints(lineSegments);

        //sort line segments based on y axis
        std::sort(lineSegmentsSortedPoints.begin(), lineSegmentsSortedPoints.end(), comp);



        //group into lines
        vector<Line> lines = groupLineSegments(lineSegmentsSortedPoints);

        cv::Mat coloredLines = drawColouredLines(cut, lines);
        imshow("lines grouped", coloredLines);


        vector<cv::Vec4i> rightLaneLineSegmentsOld = rightLaneLineSegments;
        vector<cv::Vec4i> leftLaneLineSegmentsOld = leftLaneLineSegments;
        vector<cv::Vec4i> rightLaneLineSegments, leftLaneLineSegments;

        classifyLaneLines(rightLaneLineSegments, leftLaneLineSegments, lines);


        if (rightLaneLineSegments.size() > 0){
            std::sort(rightLaneLineSegments.begin(), rightLaneLineSegments.end(), comp2);
            cv::Vec4i lastRightSegment = rightLaneLineSegments.back();
            cv::Vec4i newEndLineSegmentRight = calculateNewEndLineSegment(lastRightSegment);
            rightLaneLineSegments.push_back(newEndLineSegmentRight);
        } else{
            rightLaneLineSegments = rightLaneLineSegmentsOld;
        }

        if (leftLaneLineSegments.size() > 0){
            std::sort(leftLaneLineSegments.begin(), leftLaneLineSegments.end(), comp2);
            cv::Vec4i lastLeftSegment = leftLaneLineSegments.back();
            cv::Vec4i newEndLineSegmentLeft = calculateNewEndLineSegment(lastLeftSegment);
            leftLaneLineSegments.push_back(newEndLineSegmentLeft);
        } else{
            leftLaneLineSegments = leftLaneLineSegmentsOld;
        }


        cv::Mat lanes = drawLanes(cut, leftLaneLineSegments,rightLaneLineSegments);
        cv::imshow("lanes", lanes);

        cv::waitKey(0);










    }






}