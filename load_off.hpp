#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <glm/glm.hpp>


class OffModel {
	public:

	// Data member
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> features;
	std::vector<int> hits;
	std::vector<glm::ivec3> faces;
	glm::vec3 default_feature = glm::vec3(0.2);
	int datasetIndex;


	OffModel()
	{

	}


	// https://en.wikipedia.org/wiki/OFF_(file_format)
	OffModel(std::string path, int index)
	{
		datasetIndex = index;
		std::string line;
		std::ifstream myfile (path);
		if (not myfile.is_open())
		{
			std::cout << "Unable to open file " << path << std::endl; 
			return;
		}

		// first line
		std::getline(myfile,line);
		if (line.find("OFF") == std::string::npos)
		{
			std::cout << "File " << path << "is not an OFF model" << std::endl;
			index = -1;
			return;
		}

		// second line
		int n_vertices, n_faces, zero;
		std::getline(myfile,line);
		std::stringstream ss(line);
		ss >> n_vertices >> n_faces >> zero;
		std::cout << "Loading " << n_vertices << " and " <<n_faces << " faces" << std::endl; 
		if (zero != 0)
			std::cout << "Warning: n_edges is not zero" << std::endl;

		// all vertices
		float x,y,z;
		for(int index = 0; index < n_vertices; index++)
		{
			std::getline(myfile,line);
			std::stringstream ss(line);
			ss >> x >> y >> z;
			// vertices[index] = glm::vec3(x,y,z);
			vertices.push_back(glm::vec3(x,y,z));
			
			// important for later
			// features.push_back(glm::vec3(1.0));
			features.push_back(default_feature);
			hits.push_back(0);
		}

		// all faces
		int i, j, k, three;
		for(int index = 0; index < n_faces; index++)
		{
			std::getline(myfile,line);
			std::stringstream ss(line);
			ss >> three >> i >> j >> k;
			if (three != 3)
				std::cout << "Warning: face has more than 3 vertices?" << std::endl;
			// faces[index] = glm::ivec3(i,j,k);
			faces.push_back(glm::ivec3(i,j,k));
		}

		// provide statistics
		// #pragma omp parallel for
		auto min = vertices[0];
		auto max = vertices[0];
		for(int index = 1; index < n_vertices; index++)
		{
			min = glm::min(min, vertices[index]);
			max = glm::max(max, vertices[index]);
		}
		
		// std::cout << "Max vertex is " << max.x << " " << max.y << " " << max.z << " " << std::endl;
		// std::cout << "Min vertex is " << min.x << " " << min.y << " " << min.z << " " << std::endl;


		myfile.close();
	};
};