#include <cvsba/cvsba.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <opencv2/core/core.hpp>
#include <ctime>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    //std::cout<<"SBA"<<endl;
    std::vector< cv::Point3d > points3D;
    std::vector< std::vector< cv::Point2d > > pointsImg;
    std::vector< std::vector< int > > visibility;
    std::vector< cv::Mat > cameraMatrix, distCoeffs, R, T;
    int NPOINTS; // number of 3d points
    int NCAMS; // number of cameras
    int base;
    string data;
    ifstream infile;
    infile.open("data/camera/sba_cfg.txt");
    if (!infile.is_open())
    {
        cout << "File cannot be open!" << endl;
        exit(0);
    }
    while(infile.good())
    {
        infile>>data;
        data.erase(std::remove(data.begin(), data.end(), '"'), data.end());
        NCAMS = stoi(data);
        infile>>data;
        data.erase(std::remove(data.begin(), data.end(), '"'), data.end());
        NPOINTS = stoi(data);
        infile>>data;
        data.erase(std::remove(data.begin(), data.end(), '"'), data.end());
        base = stoi(data);
    }
    infile.close();
    //std::cout<<"SBA Configuration: Cams= "<<NCAMS<<" Points= "<<NPOINTS<< " Base Frame: "<<base<<std::endl;
    //std::cout<<"Reading threeD"<<endl;
    //3D Points
    string filename_3D = "data/camera/threeD.txt";
    infile.open(filename_3D);
    if (!infile.is_open())
    {
        std::cout << "File cannot be open!" << std::endl;
        exit(0);
    }
    points3D.resize(NPOINTS);
    int ite=0;
    while(infile.good())
    {
        infile>>data;
        data.erase(std::remove(data.begin(), data.end(), ','), data.end());
        data.erase(std::remove(data.begin(), data.end(), '['), data.end());
        data.erase(std::remove(data.begin(), data.end(), ','), data.end());
        double num1;
        num1=stof(data);
        infile>>data;
        data.erase(std::remove(data.begin(), data.end(), ','), data.end());
        data.erase(std::remove(data.begin(), data.end(), '['), data.end());
        data.erase(std::remove(data.begin(), data.end(), ','), data.end());
        double num2;
        num2=stof(data);
        infile>>data;
        data.erase(std::remove(data.begin(), data.end(), ','), data.end());
        data.erase(std::remove(data.begin(), data.end(), '['), data.end());
        data.erase(std::remove(data.begin(), data.end(), ','), data.end());
        double num3;
        num3=stof(data);
        points3D[ite] = cv::Point3d(num1, num2, num3);
        ite++;
    }
    infile.close();

    visibility.resize(NCAMS);
    //std::cout<<"Reading Visibility Matrix"<<std::endl;
    for(int i=0; i<NCAMS; i++)
    {
        visibility[i].resize(NPOINTS);
        visibility[i+1].resize(NPOINTS);
        string filename = "data/camera/visibility" + std::to_string((2*base)+i) + ".txt";
        //std::cout<<"Loading visibility for frame: "<<(2*base)+i<<std::endl;
        infile.open(filename);
        if (!infile.is_open())
        {
            cout << "File cannot be open!" << endl;
            exit(0);
        }
        int j=0;
        while(infile.good())
        {
            infile>>data;
            data.erase(std::remove(data.begin(), data.end(), ','), data.end());
            data.erase(std::remove(data.begin(), data.end(), '['), data.end());
            data.erase(std::remove(data.begin(), data.end(), ','), data.end());
            int num;
            num=stoi(data);
            visibility[i][j]=num;
            visibility[i+1][j]=num;
            j++;
        }
        infile.close();
        i+=1;
    }
    //std::cout<<"Visibility Loaded"<<std::endl;
   //Read Image Points
    pointsImg.resize(NCAMS);
    for(int i=0; i<NCAMS; i++) pointsImg[i].resize(NPOINTS);

    for(int i=0;i<NCAMS;i++)
    {

        string filename = "data/camera/cam" + std::to_string(i+(base *2)) + ".txt";
        //std::cout<<"Loading Image points from camera: "<<i+(base *2)<<std::endl;
        infile.open(filename);
        if (!infile.is_open())
        {
            cout << "File cannot be open!" << endl;
            exit(0);
        }
        ite=0;
        while(infile.good())
        {
            if(visibility[i][ite] == 1)
            {
                infile>>data;
                data.erase(std::remove(data.begin(), data.end(), ','), data.end());
                data.erase(std::remove(data.begin(), data.end(), '['), data.end());
                data.erase(std::remove(data.begin(), data.end(), ','), data.end());
                double num1;
                num1=stof(data);
                infile>>data;
                data.erase(std::remove(data.begin(), data.end(), ','), data.end());
                data.erase(std::remove(data.begin(), data.end(), '['), data.end());
                data.erase(std::remove(data.begin(), data.end(), ','), data.end());
                double num2;
                num2=stof(data);
                pointsImg[i][ite] = cv::Point2d(num1, num2);
            }

            //std::cout<<pointsImg[i][ite]<<std::endl;
            ite++;
        }
        infile.close();
    }
    //std::cout<<"All Image points Loaded"<<std::endl;

    //std::cout<<"Visibility Matrix Loaded"<<std::endl;
    //Read Camera Matrices
    cameraMatrix.resize(NCAMS);
    for(int i=0;i<NCAMS;i++)
    {
        cameraMatrix[i] = cv::Mat::eye(3,3,CV_64FC1);
        if(i == 0 || i == 1)
        {
            cameraMatrix[i].ptr<double>(0)[0] = 625.4584;
            cameraMatrix[i].ptr<double>(0)[1]=0.;
            cameraMatrix[i].ptr<double>(0)[2] = 624.535;
            cameraMatrix[i].ptr<double>(0)[3]=0.;
            cameraMatrix[i].ptr<double>(0)[4] = 625.4584;
            cameraMatrix[i].ptr<double>(0)[5] = 191.1291;
            cameraMatrix[i].ptr<double>(0)[6]=0.;
            cameraMatrix[i].ptr<double>(0)[7]=0.;
            cameraMatrix[i].ptr<double>(0)[8]=1.;
        }
        else
        {
            for(int j=0;j<9;j++)
            {
                cameraMatrix[i].ptr<double>(0)[j] = cameraMatrix[i%2].ptr<double>(0)[j];
            }
        }
    }
    //std::cout<<"Camera Matrices Loaded"<<std::endl;
    //Read Distortion params
    distCoeffs.resize(NCAMS);
    for(int i=0; i<NCAMS; i++) distCoeffs[i] = cv::Mat(5,1,CV_64FC1, cv::Scalar::all(0));
    //std::cout<<"Camera D Loaded"<<std::endl;
    //Fill Rotation
    R.resize(NCAMS);
    for(int i=0; i<NCAMS-2; i++)
    {
        R[i] = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));
        string filename = "data/camera/rotation" + std::to_string(i+(base *2)) + ".txt";
        //std::cout<<"Rotation Matrix for Camera "<<i+(base *2)<<" loaded."<<std::endl;
        FileStorage fs(filename,FileStorage::READ);
        fs["mat"] >> R[i];
        fs.release();
    }
    R[NCAMS-2] = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));
    R[NCAMS-2].ptr<double>(0)[0] = R[NCAMS-4].ptr<double>(0)[0];
    R[NCAMS-2].ptr<double>(0)[1] = R[NCAMS-4].ptr<double>(0)[1];
    R[NCAMS-2].ptr<double>(0)[2] = R[NCAMS-4].ptr<double>(0)[2];

    R[NCAMS-1] = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));
    R[NCAMS-1].ptr<double>(0)[0] = R[NCAMS-3].ptr<double>(0)[0];
    R[NCAMS-1].ptr<double>(0)[1] = R[NCAMS-3].ptr<double>(0)[1];
    R[NCAMS-1].ptr<double>(0)[2] = R[NCAMS-3].ptr<double>(0)[2];
    //Fill Translation
    T.resize(NCAMS);
    //Add condition for NCAMS == 2
    for(int i=0; i<NCAMS-2; i++)
    {
        T[i] = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));

        string filename = "data/camera/translation" + std::to_string(i+(base *2)) + ".txt";
        //std::cout<<"Translation Matrix for Camera "<<i+(base *2)<<" loaded."<<std::endl;
        FileStorage fs(filename,FileStorage::READ);
        fs["mat"] >> T[i];
        fs.release();
    }
    T[NCAMS-2] = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));
    T[NCAMS-2].ptr<double>(0)[0] = T[NCAMS-4].ptr<double>(0)[0];
    T[NCAMS-2].ptr<double>(0)[1] = T[NCAMS-4].ptr<double>(0)[1];
    T[NCAMS-2].ptr<double>(0)[2] = T[NCAMS-4].ptr<double>(0)[2];

    T[NCAMS-1] = cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0));
    T[NCAMS-1].ptr<double>(0)[0] = T[NCAMS-3].ptr<double>(0)[0];
    T[NCAMS-1].ptr<double>(0)[1] = T[NCAMS-3].ptr<double>(0)[1];
    T[NCAMS-1].ptr<double>(0)[2] = T[NCAMS-3].ptr<double>(0)[2];

    //std::cout<<"Rotation and Translation for final frames loaded."<<std::endl;
    //RUN SBA

 /****** RUN BUNDLE ADJUSTMENT ******/
    cvsba::Sba sba;
    cvsba::Sba::Params param;
    param.type = cvsba::Sba::TYPE::MOTIONSTRUCTURE;
    param.fixedIntrinsics =5;
    param.fixedDistortion =5;
    //param.iterations = 1000000;
    //param.minError = 0.0001;
    param.verbose = true;
    sba.setParams(param);
    clock_t begin = clock();
    sba.run(points3D,  pointsImg,  visibility,  cameraMatrix,  R,  T, distCoeffs);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout<<"Time Taken "<<elapsed_secs<<std::endl;
    std::cout<<"Initial error="<<sba.getInitialReprjError()<<". Final error="<<sba.getFinalReprjError()<<std::endl;

    for(int i=0;i<NCAMS;i++)
    {
        string filename = "data/camera/rotation" + std::to_string(i+(base *2)) + ".txt";
        cv::FileStorage file(filename, cv::FileStorage::WRITE);
        // Write to file!
        file << "mat" <<R[i];
        file.release();

        filename = "data/camera/translation" + std::to_string(i+(base *2)) + ".txt";
        cv::FileStorage file1(filename, cv::FileStorage::WRITE);
        // Write to file!
        file1 << "mat" << T[i];
        file1.release();
    }
    string filename = "data/results/threeD" + std::to_string(base) + ".txt";
    cv::FileStorage file(filename, cv::FileStorage::WRITE);
    // Write to file!
    file << "mat" <<points3D;
    file.release();
    return 0;
}
