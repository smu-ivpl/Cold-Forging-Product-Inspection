#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace dnn;


int main() {
	string basic_path = "D:/PROJECTS/HANSUNG/datasets/datasets_final/";

	string model_path = basic_path + "onnx/best_weight_25.onnx";
	string path = basic_path + "onnx/example/input/";

	// ====================================================image selection====================================================
	// Error test images.
	// Test image1. //ppt reference.
	string img_name = "IMG_BE710001_2022-10-07_18-56-07_Normal.png";
	string rf_name = "IMG_BE710001_2022-10-07_18-56-07_SpecularRF.png";
	// Test image2.
	// string img_name = "IMG_BE710001_2022-10-07_18-22-58_Normal.png";
	// string rf_name = "IMG_BE710001_2022-10-07_18-22-58_SpecularRF.png";
	// ======================================================================================================================


	string img_path = path + img_name;
	string rf_path = path + rf_name;

	string save_path = basic_path + "onnx/example/result/";

	float ratio = 2.5; // zoom in ratio.
	string label;
	string find_label = "_Normal.png";
	string result_label = "Normal Image!";
	string save_name = img_name.replace(img_name.find(find_label), find_label.length(), "_"); // ex. "IMG_BE710001_2022-10-07_18-18-52_"

	// deeplearning model fixed input size.
	int inputW = 256;
	int inputH = 512;
	// crop size before zoomin.
	int cropW = (int)(inputW / ratio);
	int cropH = (int)(inputH / ratio);

	// image load.
	Mat img = imread(img_path, 0);
	Mat c_img = img.clone(); // as candidate image display
	Mat i_img = img.clone(); // as imput image
	Mat rf_img = imread(rf_path, 0);
	Mat binary_img; // image for binarization.

	// Eliminate unnecessary areas.
	Rect rect(843, 0, 1024, img.rows);
	Mat srf_img = rf_img(rect);
	c_img = c_img(rect);
	i_img = i_img(rect);

	// Binarization.
	threshold(srf_img, binary_img, 200, 255, THRESH_BINARY); // The higher than 200pixel => white.
	binary_img = 255 - binary_img; // inverse color for object recognition using labling function.

	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(binary_img, labels, stats, centroids);

	if (cnt < 2) {
		label = "normal_";
		imwrite(save_path + label + img_name, img);
		cout << "This image is " << result_label << endl;
		return 0;
	}

	int num_crop = 0;

	vector<int> cx_vector, cy_vector;
	vector<int> l_vector, t_vector;
	vector<int> rm_index;
	cvtColor(srf_img, srf_img, COLOR_GRAY2BGR);


	// list saved to the labling result for the center coordinates.
	for (int i = 1; i < cnt; i++) {
		int area = stats.at<int>(i, CC_STAT_AREA);
		int left = stats.at<int>(i, CC_STAT_LEFT);
		int top = stats.at<int>(i, CC_STAT_TOP);
		int width = stats.at<int>(i, CC_STAT_WIDTH);
		int height = stats.at<int>(i, CC_STAT_HEIGHT);

		if (area < 40)
			continue;

		num_crop += 1;

		int cx = centroids.at<double>(i, 0);
		int cy = centroids.at<double>(i, 1);

		cx_vector.push_back(cx);
		cy_vector.push_back(cy);

		// line(srf_img, Point(cx, cy), Point(cx, cy), Scalar(255, 0, 255), 10);
		// rectangle(srf_img, Point(left, top), Point(left + width, top + height), Scalar(255, 0, 255), 10);
		// putText(srf_img, to_string(i), Point(cx, cy - 20), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(255, 0, 0), 5);
		// putText(srf_img, to_string(area), Point(cx, cy + 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar(255, 0, 255), 5);
	}

	// process in case out of range of image.
	for (int index = 0; index < cx_vector.size(); index++) {
		int cx = cx_vector[index];
		int cy = cy_vector[index];

		int l = (int)(cx - cropW / 2);
		int r = (int)(cx + cropW / 2);
		int t = (int)(cy - cropH / 2);
		int b = (int)(cy + cropH / 2);

		// process in case out of range of image.
		if (l < 0) {
			r -= l;
			l = 0;
		}
		else if (r > srf_img.cols) {
			l -= (r - srf_img.cols);
			r = srf_img.cols;
		}

		if (t < 0) {
			b -= t;
			t = 0;
		}
		else if (b > srf_img.rows) {
			t -= (b - srf_img.rows);
			b = srf_img.rows;
		}

		l_vector.push_back(l);
		t_vector.push_back(t);

		// find index for remove the coordinates if there is defect in one crop image.
		for (int i = 0; i < cx_vector.size(); i++) {
			if (i == index)
				continue;
			int cx_value = cx_vector[i];
			int cy_value = cy_vector[i];
			if (cx_value<r && cx_value>l && cy_value<b && cy_value>t)
				rm_index.push_back(i);
		}
	}

	// remove the duplication.
	sort(rm_index.begin(), rm_index.end());
	rm_index.erase(unique(rm_index.begin(), rm_index.end()), rm_index.end());
	// for make remove index to descending ordier.
	sort(rm_index.begin(), rm_index.end(), greater<int>());

	for (auto ri : rm_index) {
		l_vector.erase(l_vector.begin() + ri);
		t_vector.erase(t_vector.begin() + ri);
	}


	// Trained DecisionNet weight model load.
	Net net = readNet(model_path); //readNetFromONNX.

	if (net.empty()) {
		cerr << "Network load filed!" << endl;
		return -1;
	}

	// Using GPU.
	net.setPreferableBackend(DNN_BACKEND_CUDA);
	net.setPreferableTarget(DNN_TARGET_CUDA);

	// inferece using crop image as input.
	for (int li = 0; li < l_vector.size(); li++) {
		Rect rec(l_vector[li], t_vector[li], cropW, cropH);
		Mat crop_img = i_img(rec);
		imwrite(save_path + "crop/" + to_string(li) + save_name + "_crop.png", crop_img);

		resize(crop_img, crop_img, Size(inputW, inputH)); // zoom in!
		Mat inputBlob = blobFromImage(crop_img, 1.0, Size(inputW, inputH));

		net.setInput(inputBlob);
		clock_t start_t = clock();
		Mat output = net.forward();
		clock_t end_t = clock();

		////// cout << "time is " << (double)(end_t - start_t) / CLOCKS_PER_SEC << endl;

		// shape conversion from tensor to image format.
		Mat segmentation = output.reshape(1, output.size[2]);

		// for zoom in result image to input size.
		resize(segmentation, segmentation, Size(inputW, inputH));

		// result image binarization.
		Mat result;
		int th = 3;
		threshold(segmentation, result, th, 255, THRESH_BINARY);
		imwrite(save_path + "output/"+ to_string(li) + save_name + "segmentation.png", result);

		rectangle(c_img, rec, Scalar(255, 0, 255), 7);

		// create to bounding box.
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		int x_min, x_max;
		int y_min, y_max;
		vector<int> x_vector, y_vector;

		result.convertTo(result, CV_8UC1);
		findContours(result, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		cvtColor(crop_img, crop_img, COLOR_GRAY2BGR);

		if (contours.size() == 0) {
			label = "normal_";
			imwrite(save_path + "bbox/" + label + to_string(li) + save_name + "bbox.png", crop_img);
			continue;
		}
		else
			result_label = "Error Image!";


		for (int i = 0; i < contours[0].size(); i++) {
			x_vector.push_back(contours[0][i].x);
			y_vector.push_back(contours[0][i].y);
		}

		x_min = *min_element(x_vector.begin(), x_vector.end());
		x_max = *max_element(x_vector.begin(), x_vector.end());
		y_min = *min_element(y_vector.begin(), y_vector.end());
		y_max = *max_element(y_vector.begin(), y_vector.end());

		rectangle(crop_img, Point(x_min, y_min), Point(x_max, y_max), Scalar(255, 0, 0), 3);

		label = "error_";
		imwrite(save_path + "bbox/" + label + to_string(li) + save_name + "_bbox_result.png", crop_img);
		// imshow("img", crop_img);
		// waitKey(0);
	}

	imwrite(save_path + "candidate/" + save_name + "_candidate.png", c_img);

	cout << "This image is " << result_label << endl;

	return 0;
}