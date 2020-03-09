
#include <ntk/ntk.h>
#include <ntk/utils/debug.h>
#include <ntk/camera/openni2_grabber.h>

#include <QApplication>
#include <QDir>
#include <QMutex>

using namespace cv;
using namespace ntk;

namespace opt
{
ntk::arg<bool> high_resolution("--highres", "High resolution color image.", 0);
}

int main(int argc, char **argv)
{
    // Parse command line options.
    arg_base::set_help_option("-h");
    arg_parse(argc, argv);

    // Set debug level to 1.
    ntk::ntk_debug_level = 2;

    // Set current directory to application directory.
    // This is to find Nite config in config/ directory.
    QApplication app (argc, argv);
    QDir::setCurrent(QApplication::applicationDirPath());

    // Declare the global OpenNI driver. Only one can be instantiated in a program.
    Openni2Driver ni_driver;

    std::cout << "Instantiated ni_driver!" << std::endl;

    // Declare the frame grabber.
    Openni2Grabber grabber(ni_driver);

    // High resolution 1280x1024 RGB Image.
    if (opt::high_resolution())
        grabber.setHighRgbResolution(true);

    // Start the grabber.
    grabber.connectToDevice();
    ntk_info("Starting grabber\n"); 
    grabber.start();

    // Holder for the current image.
    RGBDImage image;

    // Image post processor. Compute mappings when RGB resolution is 1280x1024.
    OpenniRGBDProcessor post_processor;

    namedWindow("depth");
    namedWindow("color");
//    namedWindow("users");

    ntk_info("Entering grab loop\n"); 
    char last_c = 0;
    while (true && (last_c != 27))
    {
        // Wait for a new frame, get a local copy and postprocess it.
        ntk_info("Waiting for frame\n"); 
        grabber.waitForNextFrame();
        ntk_info("Copying image\n"); 
        grabber.copyImageTo(image);
        ntk_info("Processing image\n"); 
        post_processor.processImage(image);

        // Prepare the depth view, mapped onto rgb frame.
        ntk_info("Normalizing depth\n"); 
        cv::Mat1b debug_depth_img = normalize_toMat1b(image.mappedDepth());

        // Prepare the color view with skeleton and handpoint.
        cv::Mat3b debug_color_img;
        ntk_info("copying rgb\n"); 
        image.rgb().copyTo(debug_color_img);

        // Prepare the user mask view as colors.
        cv::Mat3b debug_users;
        image.fillRgbFromUserLabels(debug_users);

        ntk_info("showing depth\n"); 
        imshow("depth", debug_depth_img);
        ntk_info("showing color\n"); 
        imshow("color", debug_color_img);
        ntk_info("showing debug\n"); 
//        imshow("users", debug_users);
        last_c = (cv::waitKey(10) & 0xff);
    }
    grabber.stop();
}
