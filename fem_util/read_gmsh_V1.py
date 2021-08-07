def read_gmsh(file_name):

    import numpy as np


    # Element types
    # 1 2-node line.
    # 2 3-node triangle.
    # 3 4-node quadrangle.
    # 4 4-node tetrahedron.
    # 5 8-node hexahedron.
    # 6 6-node prism.
    # 7 5-node pyramid.
    # 8 3-node second order line (2 nodes associated with the vertices and 1 with the edge).
    # 9 6-node second order triangle (3 nodes associated with the vertices and 3 with the edges).
    # 10 9-node second order quadrangle (4 nodes associated with the vertices, 4 with the edges and 1 with the face).
    # 11 10-node second order tetrahedron (4 nodes associated with the vertices and 6 with the edges).
    # 12 27-node second order hexahedron (8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume).
    # 13 18-node second order prism (6 nodes associated with the vertices, 9 with the edges and 3 with the quadrangular faces).
    # 14 14-node second order pyramid (5 nodes associated with the vertices, 8 with the edges and 1 with the quadrangular face).
    # 15 1-node point.
    # 16 8-node second order quadrangle (4 nodes associated with the vertices and 4 with the edges).
    # 17 20-node second order hexahedron (8 nodes associated with the vertices and 12 with the edges).
    # 18 15-node second order prism (6 nodes associated with the vertices and 9 with the edges).
    # 19 13-node second order pyramid (5 nodes associated with the vertices and 8 with the edges).
    # 20 9-node third order incomplete triangle (3 nodes associated with the vertices, 6 with the edges)
    # 21 10-node third order triangle (3 nodes associated with the vertices, 6 with the edges, 1 with the face)
    # 22 12-node fourth order incomplete triangle (3 nodes associated with the vertices, 9 with the edges)
    # 23 15-node fourth order triangle (3 nodes associated with the vertices, 9 with the edges, 3 with the face)
    # 24 15-node fifth order incomplete triangle (3 nodes associated with the vertices, 12 with the edges)
    # 25 21-node fifth order complete triangle (3 nodes associated with the vertices, 12 with the edges, 6 with the face)
    # 26 4-node third order edge (2 nodes associated with the vertices, 2 internal to the edge)
    # 27 5-node fourth order edge (2 nodes associated with the vertices, 3 internal to the edge)
    # 28 6-node fifth order edge (2 nodes associated with the vertices, 4 internal to the edge)
    # 29 20-node third order tetrahedron (4 nodes associated with the vertices, 12 with the edges, 4 with the faces)
    # 30 35-node fourth order tetrahedron (4 nodes associated with the vertices, 18 with the edges, 12 with the faces, 1 in the volume)
    # 31 56-node fifth order tetrahedron (4 nodes associated with the vertices, 24 with the edges, 24 with the faces, 4 in the volume)  

    nodel = [2,3,4,4,8,6,5,3,6,9,10,27,18,14,1,8,20,15,13,9,10,12,15,15,21,4,5,6,20,35,56]
    
    # Create a structure as in matlab for the output
    class structtype():
        pass
    out = structtype()

    f = open(file_name)
    f.readline() # '$MeshFormat\n'
    f.readline() # '2.2 0 8\n'
    f.readline() # '$EndMeshFormat\n'
    f.readline() # '$Nodes\n'
    
    
    out.maxnp = int(f.readline()) # '8\n'
    cord = np.fromfile(f,count = out.maxnp*4, sep=" ").reshape((out.maxnp,4))
    out.cord = np.delete(cord,0,1) #delete 0th col in coord array

    f.readline() # '$EndNodes\n'
    f.readline() # '$Elements\n'
    maxel = int(f.readline()) # '2\n'
    ind = np.zeros([1,len(nodel)])


    out.nop = [ [], [] ] # create cell for line and triangle element indices
    out.nb = [ [], [] ]
    out.nodel = [ [], [] ]
    out.phys_group = [ [], [] ]
    out.elem_ent = [ [], [] ]
    out.maxel = [ [], [] ]

    for i in range(0,maxel):
        luvut = f.readline()
        luvut = [int(s) for s in luvut.split(' ')]
        typek = luvut[1]
        ntag = luvut[2]
        tags = luvut[3: 3+int(ntag)]
        nodelk = nodel[typek-1]
        nodes = luvut[3+ntag:3+ntag+nodelk]
        ind[0,typek-1] = ind[0,typek-1] + 1
    
        out.nop[typek-1].append(nodes)  
        out.nb[typek-1].append(i)
        out.nodel[typek-1].append(nodelk)
        if ntag == 2:
            out.phys_group[typek-1].append(tags[0])
            out.elem_ent[typek-1].append(tags[1])
            out.maxel[typek-1] = (ind[0,typek-1])
#elems = np.fromfile(f,sep=" ") # $EndElements read as -1
    return out
