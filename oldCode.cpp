vector<cv::Vec4i> removeInterceptingLineSegs(vector<cv::Vec4i> lineSegments){
    /// <summary>remove overlapping line segments</summary>
    vector<cv::Vec4i> thinnedLines = vector<cv::Vec4i>(lineSegments);
    auto it = thinnedLines.begin();
    while (it != thinnedLines.end()) {
        cv::Vec4i line1 = *it;
        cv::Point l1o, l1d;
        l1o = cv::Point(line1[0], line1[1]);
        l1d = cv::Point(line1[2], line1[3]);

        auto it2 = it + 1;
        bool remove = false;
        cv::Point l2o, l2d;
        while(it2 != thinnedLines.end()) {
            cv::Vec4i line2 = *it2;
            //if going through list multiple times later be sure of ordering again;
            l2o = cv::Point(line2[0], line2[1]);
            l2d = cv::Point(line2[2], line2[3]);
            cv::Point intersection;

            if ((findIntersection(intersection, l1o, l1d, l2o, l2d)) and (abs(angle(l1o, l1d) - angle(l2o, l2d)) < 15) and (intersection.y > l2d.y and intersection.y < l2o.y) and (intersection.y > l1d.y and intersection.y < l2o.y)){
                remove = true;
                cout << "merge" << endl;
                break;

            } else {
                ++it2;
            }
        }

        if(remove) {
            //erase only if more following
            cv::Vec4i newLine = mergeLines(l1o, l1d, l2o, l2d);
            it2 = thinnedLines.erase(it2);
            it = thinnedLines.erase(it);
            thinnedLines.insert(thinnedLines.begin(), newLine);
        }
        else {
            ++it;
        }
    }
    return thinnedLines;
}


vector<cv::Vec4i> removeInterceptingLineSegs(vector<cv::Vec4i> lineSegments){
    /// <summary>remove overlapping line segments</summary>
    vector<cv::Vec4i> thinnedLines = vector<cv::Vec4i>(lineSegments);
    auto it = thinnedLines.begin();
    while (it != thinnedLines.end()) {
        cv::Vec4i line1 = *it;
        cv::Point l1o, l1d;
        l1o = cv::Point(line1[0], line1[1]);
        l1d = cv::Point(line1[2], line1[3]);

        auto it2 = it + 1;
        bool remove = false;
        cv::Point l2o, l2d;

        while(it2 != thinnedLines.end()) {
            cv::Vec4i line2 = *it2;
            //if going through list multiple times later be sure of ordering again;
            l2o = cv::Point(line2[0], line2[1]);
            l2d = cv::Point(line2[2], line2[3]);

            cv::Point intersection;
            if (overlap(l1o, l1d, l2o, l2d) and (abs(angle(l1o, l1d) - angle(l2o, l2d)) < 15)){
                remove = true;
                break;

            } else {
                ++it2;
            }
        }

        if(remove) {
            //erase only if more following
            cv::Vec4i newLine = mergeLines(l1o, l1d, l2o, l2d);
            it2 = thinnedLines.erase(it2);
            it = thinnedLines.erase(it);
            thinnedLines.insert(thinnedLines.begin(), newLine);
        }
        else {
            ++it;
        }
    }
    return thinnedLines;
}




#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <unistd.h>
#include <string>



void Images2Video(string folderName) {
    int frame_width = 752;
    int frame_height = 480;

    //nt codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    int codec = cv::VideoWriter::fourcc('W', 'M', 'V', '2');
    cv::VideoWriter video("out.wmv", codec, 10, cv::Size(frame_width, frame_height), false);

    vector<cv::String> filenames;

    string fpath = "footage_collection" + OS_SEP + folderName +  OS_SEP + "*.png";
    cv::glob(fpath, filenames, false);
    if (filenames.empty()){
        cout << "Could not find images" << endl;
    }
    size_t count = filenames.size();

    for (size_t i = 0; i < count; i++) {
        cout << filenames[i] << endl;
        cv::Mat frame = imread(filenames[i], cv::IMREAD_GRAYSCALE);
        video.write(frame);

        imshow("image", frame);
        cv::waitKey(10);
    }
    video.release();

    //play video
    cv::VideoCapture cap("out.wmv");
    if ( !cap.isOpened() ){
        cout << "Cannot open the video file. \n";
    }

    namedWindow("vid",cv::WINDOW_AUTOSIZE); //create a window called "MyVideo"

    while(1){
        cv::Mat frame;
        if (!cap.read(frame)) {
            cout<<"\n Cannot read the video file. \n";
            break;
        }
        imshow("frame", frame);
        if(cv::waitKey(30) == 27) {
            break;
        }
    }
}


void playImages(string folderName){
    vector<cv::String> fn;
    string fpath = "footage_collection" + OS_SEP + folderName +  OS_SEP + "*.png";

    cv::glob(fpath, fn, false);
    if (fn.empty()){
        cout << "Could not find images" << endl;
    }
    size_t count = fn.size();

    for (size_t i = 0; i < count; i++) {
        cout << fn[i] << endl;
        cv::Mat frame = imread(fn[i], cv::IMREAD_GRAYSCALE);

        imshow("image", frame);
        cv::waitKey(10);
    }
}


cv::Mat roi(cv::Mat img){
    int height = int(img.rows);
    int width = int(img.cols);
    int centerX = int(width / 2);

    int roiOffsetX = centerX - 200;
    int roiOffsetY = height - 200;
    int roiHeight = 190;
    int roiWidth = 300;

    cv::Mat roi = img(cv::Rect(roiOffsetX, roiOffsetY, roiWidth, roiHeight));
    return roi;
}


cv::Mat roi2(cv::Mat img) {
    int height = int(img.rows);
    int width = int(img.cols);
    int centerX = int(width / 2);
    int roiOffsetX = centerX - 150;
    int roiOffsetY = height - 400;

    int roiHeight = 600 ;
    int roiWidth = 300;
    cv::Rect roi = cv::Rect(roiOffsetX, roiOffsetY, roiWidth, roiHeight);

    int maskColour = 255;
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    rectangle(mask, roi, cv::Scalar(255), cv::FILLED);
    //cv::imshow("mask", mask);
    //waitKey(0);

    cv::Mat result = cv::Mat(img.rows, img.cols, CV_8UC1, img.type());
    result.setTo(cv::Scalar(0, 0, 0));

    img.copyTo(result, mask);
    //cv::imshow("result", result);
    //waitKey(0);
    return result;
}


void morphologicalOps(Mat cannyImg){
    // morphological operations to get better and fewer line segments;
    // too much information loss!!!
    Mat dilated;
    Mat eroded;

    Mat kernel = getStructuringElement (MORPH_ELLIPSE, Point(9, 9));
    dilate(cannyImg, dilated, kernel);

    Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Point(12, 12));
    erode(dilated, eroded, kernel1);
}


//        vector<Vec4i> linesCutAngles = vector<Vec4i>(sortedLines);
//        int avgAngle = 0;
//
//        sortedLines
//            avgAngle += angle(Point(line[0], line[1]), Point(line[2], line[3]));
//        }
//
//        avgAngle = avgAngle / linesCutAngles.size();
//        cout << "angle:" << avgAngle << endl;
//        for (auto it = linesCutAngles.begin(); it != linesCutAngles.end();){
//            Vec4i line = *it;
//            if((abs(avgAngle - angle(Point(line[0], line[1]), Point(line[2], line[3]))) >  60)){
//                it = linesCutAngles.erase(it);
//            }
//            else{
//                ++it;
//            }
//
//        }


//        //vector<Vec4i> lineClusters[sortedLines.size()];
//        vector<vector<Vec4i>> lineClusters;
//        for(Vec4i line1: sortedLines){
//            Point l1o = Point(line1[0], line1[1]);
//            Point l1d = Point(line1[2], line1[3]);
//            vector<Vec4i> cluster;
//            for(Vec4i line2: sortedLines){
//                Point l2o = Point(line2[0], line2[1]);
//                Point l2d = Point(line2[2], line2[3]);
//                if ((getDistance(l1o, l1d, l2o, l2d) <= 20) and
//                    (abs(angle(l1o, l1d) - angle(l2o, l2d)) <= 10)) {
//                    cluster.push_back(line2);
//                }
//            }
//            lineClusters.push_back(cluster);
//        }



//        vector<Vec4i> linesCutAnglesThinned = vector<Vec4i>(thinnedLines);
//        int avgAngleThinned = 0;
//
//        for(Vec4i line: thinnedLines){
//            avgAngleThinned     += angle(Point(line[0], line[1]), Point(line[2], line[3]));
//        }
//
//        avgAngleThinned = avgAngleThinned / linesCutAnglesThinned.size();
//        cout << "angle:" << avgAngleThinned << endl;
//        for (auto it = linesCutAnglesThinned.begin(); it != linesCutAnglesThinned.end();){
//            Vec4i line = *it;
//            if((abs(avgAngleThinned - angle(Point(line[0], line[1]), Point(line[2], line[3]))) >  60)){
//                it = linesCutAnglesThinned.erase(it);
//            }
//            else{
//                ++it;
//            }
//        }





//std::vector<int> labels;
//int equilavenceClassesCount = cv::partition(lines, labels, isEqual);

//RNG rng(215526);
//std::vector<Scalar> colors(equilavenceClassesCount);
//for (int i = 0; i < equilavenceClassesCount; i++){
//   colors[i] = Scalar(rng.uniform(30,255), rng.uniform(30, 255), rng.uniform(30, 255));;
//}


//Mat groupedLines(cut.size(), CV_8UC3);
//for (int i = 0; i < lines.size(); i++){
//    Vec4i& line = lines[i];
//    cv::line(groupedLines, Point(line[0], line[1]), Point(line[2], line[3]), colors[labels[i]], 2);
//}


//Mat coloredLinesThinned = drawLines(cut, thinnedLines);
//        Mat coloredLinesSorted = drawLines(cut, sortedLines);
//        //Mat coloredLinesCutAngles = drawLines(cut, linesCutAngles);
//        //Mat coloredLinesCutAnglesThinned= drawLines(cut, linesCutAnglesThinned);
//
//        //imshow("test", coloredLinesTest);
//
//        //imshow("image", frame);
//        //imshow("smoothed", smoothed);
//        //imshow("canny", cannyImage);
//        //imshow("cut", cut);
//        //imshow("lines", coloredLinesSorted);
//        //imshow("lines cut", coloredLinescut2);
//
//        //imshow("lines anglescut", coloredLinesCutAngles);
//        //imshow("lines anglescutThinned", coloredLinesCutAnglesThinned);
//        cout << sortedLines.size() << endl;
//       // cout << thinnedLines.size() << endl;
//        //imshow("linesEroded", coloredLinesEroded);
//        //imshow("linesThinned", coloredLinesThinned);
//        //cout << thinnedLines.size() << endl;
//        waitKey(0);




double medianMat(cv::Mat Input){
// very slightly modified Version of: https://stackoverflow.com/questions/30078756/super-fast-median-of-matrix-in-opencv-as-fast-as-matlab
// COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
    const int nVals = 4096;
    float range[] = { 0, nVals };
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    cv::Mat hist;
    calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange, uniform, accumulate);

// COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
    cv::Mat cdf;
    hist.copyTo(cdf);
    for (int i = 1; i <= nVals-1; i++){
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }
    cdf /= Input.total();

// COMPUTE MEDIAN
    double medianVal;
    for (int i = 0; i <= nVals-1; i++){
        if (cdf.at<float>(i) >= 0.5) { medianVal = i;  break; }
    }
    return medianVal/nVals; }


Mat auto_canny(Mat img){
    /// c++ implementation of procedure suggested here: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    double sigma = 0.33;
    double  median  = medianMat(img);
    int lower  = int(std::max(0.0, (1.0 - sigma) * median));
    int upper = upper = int(std::min(255.0, (1.0 + sigma) * median));
    Mat edged = canny(img, lower, upper);
    return edged;
}

int main() {
    Mat image = imread("test.png", IMREAD_GRAYSCALE);

    Mat cannyImage = canny(frame);
    Mat autoCannyImg = auto_canny(frame);

    imshow("canny", cannyImage);
    imshow("auto canny", autoCannyImg);
}