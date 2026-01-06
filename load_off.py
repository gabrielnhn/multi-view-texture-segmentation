import numpy as np

class OffModel:

    default_feature = np.array((0.2,0.2,0.2))

    # https:#en.wikipedia.org/wiki/OFF_(file_format)
    # OffModel(std::string path, int index)
    def __init__(self, path, index):

        # Data member
        self.vertices = []
        self.features = []
        self.hits = []
        self.faces = []


        datasetIndex = 0


        datasetIndex = index
        # line
        
        with open(path, "r") as myfile:

            # first line
            # std::getline(myfile,line)
            line = myfile.readline()
            # print(line)


            if (line.find("OFF") == -1):
                print(f"File {path} is not an OFF model"  )
                index = -1
                return

            # second line
            n_vertices, n_faces, zero = 0,0,0
            line = myfile.readline()
            # print(line)
            

            # std::stringstream ss(line)
            # ss >> n_vertices >> n_faces >> zero
            
            n_vertices, n_faces, zero = [int(i) for i in line.split()]
            
            
            print(f"Loading   {n_vertices} and {n_faces} faces"  ) 
            if (zero != 0):
                print( "Warning: n_edges is not zero"  )

            # all vertices
            x,y,z = 0.0,0.0,0.0

            for index in range(0, n_vertices):
            
                # std::getline(myfile,line)
                line = myfile.readline()
                # print(line)
                
                # std::stringstream ss(line)
                # ss >> x >> y >> z
                x, y, z = [float(f) for f in line.split()]


                # vertices[index] = glm::vec3(x,y,z)
                # vertices.push_back(glm::vec3(x,y,z))
                self.vertices.append((x,y,z))
                
                # important for later
                # features.push_back(glm::vec3(1.0))
                self.features.append(self.default_feature)
                self.hits.append(0)
            

            # all faces
            # i, j, k, three = 0
            # for(int index = 0 index < n_faces index++)
            for index in range(0, n_faces):
                # std::getline(myfile,line)
                # std::stringstream ss(line)
                # ss >> three >> i >> j >> k
                line = myfile.readline()
                
                three, i, j, k = [int(value) for value in line.split()]


                if (three != 3):
                    print(  "Warning: face has more than 3 vertices?"  )
                # faces[index] = glm::ivec3(i,j,k)
                # faces.push_back(glm::ivec3(i,j,k))

                self.faces.append((i,j,k))
            

            # provide statistics
            # #pragma omp parallel for
            # min = self.vertices[0]
            # max = self.vertices[0]
            # # for(int index = 1 index < n_vertices index++):
            
            
            #     min = glm::min(min, vertices[index])
            #     max = glm::max(max, vertices[index])
            
            
            # print(  "Max vertex is "  max.x  " "  max.y  " "  max.z  " "  )
            # print(  "Min vertex is "  min.x  " "  min.y  " "  min.z  " "  )


            # myfile.close()