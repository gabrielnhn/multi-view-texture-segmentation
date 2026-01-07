import numpy as np

class OffModel:
    # https:#en.wikipedia.org/wiki/OFF_(file_format)

    default_feature = np.array((0.2,0.2,0.2))
    def __init__(self, path, index):

        # Data member
        self.vertices = []
        self.features = []
        self.hits = []
        self.faces = []
        # line
        
        with open(path, "r") as myfile:
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
            n_vertices, n_faces, zero = [int(i) for i in line.split()]
            
            print(f"Loading   {n_vertices} and {n_faces} faces"  ) 
            if (zero != 0):
                print( "Warning: n_edges is not zero"  )

            # all vertices
            x,y,z = 0.0,0.0,0.0

            for index in range(0, n_vertices):
                line = myfile.readline()
                # print(line)
                
                x, y, z = [float(f) for f in line.split()]
                self.vertices.append((x,y,z))
                
                # important for later
                self.features.append(self.default_feature)
                self.hits.append(0)
            
            # all faces
            for index in range(0, n_faces):
                line = myfile.readline()
                
                three, i, j, k = [int(value) for value in line.split()]


                if (three != 3):
                    print("Warning: face has more than 3 vertices?")
            
                self.faces.append((i,j,k))
            