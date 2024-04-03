#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/search/impl/search.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

py::array_t<float> estimate_normal(py::array_t<float> pc, double normal_r)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->points.resize(pc.shape(0));
	float *pc_ptr = (float*)pc.request().ptr;
	for (int i = 0; i < pc.shape(0); ++i)
    {
		std::copy(pc_ptr, pc_ptr + 3, &cloud->points[i].data[0]);
		// std::cout << cloud->points[i] << std::endl;
		pc_ptr += 3;
    }
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	// normalEstimation.setKSearch(40);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
	normalEstimation.setSearchMethod(kdtree);

	normalEstimation.setRadiusSearch(normal_r);
	normalEstimation.compute(*normals);

	auto result = py::array_t<float>(normals->points.size() * 3);
    auto buf = result.request();
    float *ptr = (float*)buf.ptr;
	for (int i = 0; i < normals->points.size(); ++i)
    {
		std::copy(&normals->points[i].normal[0], &normals->points[i].normal[3], &ptr[i * 3]);
    }
    return result;
}


std::vector<py::array_t<float>> compute(py::array_t<float> pc, double normal_r, double shot_r)
{
	// Object for storing the point cloud.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->points.resize(pc.shape(0));
	float *pc_ptr = (float*)pc.request().ptr;
	for (int i = 0; i < pc.shape(0); ++i)
    {
		std::copy(pc_ptr, pc_ptr + 3, &cloud->points[i].data[0]);
		pc_ptr += 3;
    }

	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the SHOT descriptors for each point.
	pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>());

	// Note: you would usually perform downsampling now. It has been omitted here
	// for simplicity, but be aware that computation can take a long time.

	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setRadiusSearch(normal_r);
	// normalEstimation.setKSearch(40);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	auto result_normal = py::array_t<float>(normals->points.size() * 3);
    auto buf_normal = result_normal.request();
    float *ptr_normal = (float*)buf_normal.ptr;
	for (int i = 0; i < normals->points.size(); ++i)
    {
		std::copy(&normals->points[i].normal[0], &normals->points[i].normal[3], &ptr_normal[i * 3]);
    }
	// SHOT estimation object.
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
	shot.setInputCloud(cloud);
	shot.setInputNormals(normals);
	// The radius that defines which of the keypoint's neighbors are described.
	// If too large, there may be clutter, and if too small, not enough points may be found.
	shot.setRadiusSearch(shot_r);
//    shot.setKSearch(40);
	shot.compute(*descriptors);

	auto result = py::array_t<float>(descriptors->points.size() * 352);
    auto buf = result.request();
    float *ptr = (float*)buf.ptr;
    
    for (int i = 0; i < descriptors->points.size(); ++i)
    {
		std::copy(&descriptors->points[i].descriptor[0], &descriptors->points[i].descriptor[352], &ptr[i * 352]);
    }
    return {result, result_normal};
}

py::array_t<float> compute_color(py::array_t<float> pc, py::array_t<float> pc_color, double normal_r, double shot_r)
{
	// Object for storing the point cloud.
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->points.resize(pc.shape(0));
	float *pc_ptr = (float*)pc.request().ptr;
	float *color_ptr = (float*)pc_color.request().ptr;
	for (int i = 0; i < pc.shape(0); ++i)
    {
		cloud->points[i].x = *pc_ptr;
		cloud->points[i].y = *(pc_ptr + 1);
		cloud->points[i].z = *(pc_ptr + 2);

		uint8_t r = (*color_ptr) * 255.f;
		uint8_t g = (*(color_ptr + 1)) * 255.f;
		uint8_t b = (*(color_ptr + 2)) * 255.f;
		uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
		cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);
		// std::copy(pc_ptr, pc_ptr + 3, &cloud->points[i].data[0]);
		// std::copy(color_ptr, color_ptr + 3, &cloud->points[i].data[3]);
		pc_ptr += 3;
		color_ptr += 3;
    }

	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the SHOT descriptors for each point.
	pcl::PointCloud<pcl::SHOT1344>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT1344>());

	// Note: you would usually perform downsampling now. It has been omitted here
	// for simplicity, but be aware that computation can take a long time.

	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setRadiusSearch(normal_r);
	// normalEstimation.setKSearch(40);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	// SHOT estimation object.
	pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot;
	shot.setInputCloud(cloud);
	shot.setInputNormals(normals);
	// The radius that defines which of the keypoint's neighbors are described.
	// If too large, there may be clutter, and if too small, not enough points may be found.
	shot.setRadiusSearch(shot_r);
	shot.compute(*descriptors);

	auto result = py::array_t<float>(descriptors->points.size() * 1344);
    auto buf = result.request();
    float *ptr = (float*)buf.ptr;
    
    for (int i = 0; i < descriptors->points.size(); ++i)
    {
		std::copy(&descriptors->points[i].descriptor[0], &descriptors->points[i].descriptor[1344], &ptr[i * 1344]);
    }
    return result;
}


PYBIND11_MODULE(shot, m) {
	pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
    m.def("compute", &compute, py::arg("pc"), py::arg("normal_r")=0.1, py::arg("shot_r")=0.17);
	m.def("compute_color", &compute_color, py::arg("pc"), py::arg("pc_color"), py::arg("normal_r")=0.1, py::arg("shot_r")=0.17);
	m.def("estimate_normal", &estimate_normal, py::arg("pc"), py::arg("normal_r")=0.1);
}
