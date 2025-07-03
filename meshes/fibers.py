import h5py
import matplotlib.pyplot as plt
import unittest
import numpy as np
import pandas as pd

scar_filename = "/itf-fi-ml/home/adamjak/heart/PINNs/meshes/scarmesh_1.h5"
base_filename = "basemesh.h5"
class Test_conductivities(unittest.TestCase):
    def test_conductivities(self):
        R_i=np.matrix([[1,2],[-2,1]])
        sigma_i=np.matrix([[1,0],[0,1]])
        sigma_e=np.matrix([[2,0],[0,2]])
        sigma_i_tilde=R_i@sigma_i@R_i.T
        sigma_e_tilde=R_i@sigma_e@R_i.T
        sigma_m_tilde=sigma_i_tilde@sigma_e_tilde@np.linalg.inv(sigma_i_tilde+sigma_e_tilde)
        print(sigma_m_tilde)
        expected=np.matrix([[10/3,0],[0,10/3]])
        np.testing.assert_array_equal(sigma_m_tilde,expected)
""" 
if __name__ == "__main__":
    unittest.main()
"""

with h5py.File(base_filename, "r") as base_file:
    coordinates = base_file["coordinates"][:]
    triangles = base_file["topology"][:]
    fibres = base_file["fibres"][:]



#plt.scatter(coordinates[:, 0], coordinates[:, 1])
#plt.scatter(coordinates[30058, 0], coordinates[30058, 1])
#plt.show()
#organise in dataframe like t(triangle), v1,v2,v3(vertices)
print("triangles shape", triangles.shape)
print("coordinates shape", coordinates.shape)
print("fibres shape", fibres.shape)
df = pd.DataFrame(triangles, columns=['v1', 'v2', 'v3'])
df['t'] = df.index
df = df[['t', 'v1', 'v2', 'v3']]
print("df shape",df.shape)
#Find triangles which contain target vertex

target_vertex = 0
df_target = df.loc[(df['v1'] == target_vertex) | (df['v2'] == target_vertex) | (df['v3'] == target_vertex)]
print(df_target)
print(df_target["t"].values[0])



#average the fibre fields by triangle area
#e_f (point) = sum_(neighbouring triangles i) area(triangle_i) e_f(triangle_i)
ef_points = np.zeros((len(coordinates[:,0]), 2))
conductivities=np.zeros((len(coordinates[:,0]), 2,2))
#sigma_i=np.matrix('2.8 0;0 0.26')
#sigma_e=np.matrix('2.2 0;0 1.3')
sigma_i=np.matrix('1 0;0 1')
sigma_e=np.matrix('1 0;0 1')

n_points=len(coordinates[:,0])

for i in range(n_points):
    dt_target=df.loc[(df['v1'] == i) | (df['v2'] == i) | (df_target['v3'] == i)]
    #print(len(dt_target))
    if len(dt_target)>0:
        for j in range(len(dt_target)):
            

            x1=coordinates[dt_target["v1"].values[j],0]
            x2=coordinates[dt_target["v2"].values[j],0]
            x3=coordinates[dt_target["v3"].values[j],0]
            y1=coordinates[dt_target["v1"].values[j],1]
            y2=coordinates[dt_target["v2"].values[j],1]
            y3=coordinates[dt_target["v3"].values[j],1]
            
            area_tri= 0.5*np.abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
            
            
            ef_points[i,:]+=area_tri*fibres[dt_target["t"].values[j],:2]
    else:
        #if the target vertex is within scar tissue, it may not have neighbouring triangles
        x1=coordinates[df["v1"].values[i],0]
        x2=coordinates[df["v2"].values[i],0]
        x3=coordinates[df["v3"].values[i],0]
        y1=coordinates[df["v1"].values[i],1]
        y2=coordinates[df["v2"].values[i],1]
        y3=coordinates[df["v3"].values[i],1]
        #print("x1", x1)
        #print("x2", x2)
        #print("x3", x3)
        #print("y1", y1)
        #print("y2", y2)
        #print("y3", y3)
        
        area_tri= 0.5*np.abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
        #print("area_tri", area_tri)
        
        
        ef_points[i,:]=area_tri*fibres[i,:2]  

    norm=np.sqrt(ef_points[i,0]**2+ef_points[i,1]**2)
    ef_points[i,:]/=norm
    
    R_i=np.matrix([[ef_points[i,0], ef_points[i,1]],[-ef_points[i,1], ef_points[i,0]]])
    sigma_i_tilde=R_i@sigma_i@R_i.T
    sigma_e_tilde=R_i@sigma_e@R_i.T
    sigma_m_tilde=sigma_i_tilde@sigma_e_tilde@np.linalg.inv(sigma_i_tilde+sigma_e_tilde)
    #print(i, "shape sigma_m_tilde", sigma_m_tilde.shape)
    #print(sigma_m_tilde)
    conductivities[i,:,:]=sigma_m_tilde
    #print("conductivities[i,:,:]", conductivities[i,:,:])
    print(f'{i}/{n_points}     ',conductivities[i,:], norm, ef_points[i,:])

#if any values are nan print the index
if np.isnan(conductivities).any():
    print("Found nan in index:",np.argwhere(np.isnan(conductivities)))
#save conductivities as npy file
np.save("conductivities_1_1_1_1.npy",conductivities )
print("conductivities shape", conductivities.shape)





    


