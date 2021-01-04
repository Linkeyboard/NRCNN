import numpy as np


def yuv_import(filename, width, height, numfrm, startfrm=0):
    # Open the file
    f = open(filename, "rb")

    # Skip some frames
    luma_size = height * width
    chroma_size = luma_size // 4
    frame_size = luma_size * 3 // 2
    f.seek(frame_size * startfrm, 0)

    # Define the YUV buffer
    Y = np.zeros([numfrm, height, width], dtype=np.uint8)
    U = np.zeros([numfrm, height//2, width//2], dtype=np.uint8)
    V = np.zeros([numfrm, height//2, width//2], dtype=np.uint8)

    # Loop over the frames
    for i in range(numfrm):
        # Read the Y component
        Y[i, :, :] = np.fromfile(f, dtype=np.uint8, count=luma_size).reshape([height, width])
        # Read the U component
        U[i, :, :] = np.fromfile(f, dtype=np.uint8, count=chroma_size).reshape([height//2, width//2])
        # Read the V component
        V[i, :, :] = np.fromfile(f, dtype=np.uint8, count=chroma_size).reshape([height//2, width//2])

    # Close the file
    f.close()

    return Y, U, V

def yuv_10bit_import(filename, width, height, numfrm, startfrm=0):
    # Open the file
    f = open(filename, "rb")

    # Skip some frames
    luma_size = height * width
    chroma_size = luma_size // 4
    frame_size = luma_size * 3 // 2
    f.seek(frame_size * startfrm, 0)

    # Define the YUV buffer
    Y = np.zeros([numfrm, height, width], dtype=np.uint16)
    U = np.zeros([numfrm, height//2, width//2], dtype=np.uint16)
    V = np.zeros([numfrm, height//2, width//2], dtype=np.uint16)

    # Loop over the frames
    for i in range(numfrm):
        # Read the Y component
        Y[i, :, :] = np.fromfile(f, dtype=np.uint16, count=luma_size).reshape([height, width])
        # Read the U component
        U[i, :, :] = np.fromfile(f, dtype=np.uint16, count=chroma_size).reshape([height//2, width//2])
        # Read the V component
        V[i, :, :] = np.fromfile(f, dtype=np.uint16, count=chroma_size).reshape([height//2, width//2])

    # Close the file
    f.close()

    return Y, U, V

def yuv_export(filename, Y, U, V, skip=1):
    yfrm = Y.shape[0]
    ufrm = U.shape[0]
    vfrm = V.shape[0]

    if yfrm == ufrm == vfrm:
        numfrm = yfrm
    else:
        raise Exception("The length of the frames does not match.")

    with open(filename, "wb") as f:
        for i in range(numfrm):
            if i % skip == 0:
                f.write(Y[i, :, :].tobytes())
                f.write(U[i, :, :].tobytes())
                f.write(V[i, :, :].tobytes())

def yuv_10bit_export(filename, Y, U, V, skip=1):
    yfrm = Y.shape[0]
    ufrm = U.shape[0]
    vfrm = V.shape[0]

    if yfrm == ufrm == vfrm:
        numfrm = yfrm
    else:
        raise Exception("The length of the frames does not match.")

    with open(filename, "wb") as f:
        for i in range(numfrm):
            if i % skip == 0:
                f.write(Y[i, :, :].tobytes())
                f.write(U[i, :, :].tobytes())
                f.write(V[i, :, :].tobytes())
