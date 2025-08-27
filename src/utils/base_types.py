from nptyping import Float, NDArray, Shape, SignedInteger

ArrayNxM = NDArray[Shape["N, M"], Float] | NDArray[Shape["N, M"], SignedInteger]
ArrayNxN = NDArray[Shape["N, N"], Float] | NDArray[Shape["N, M"], SignedInteger]
Array1xN = NDArray[Shape["1, N"], Float] | NDArray[Shape["N, M"], SignedInteger]
ArrayNx1 = NDArray[Shape["N, 1"], Float] | NDArray[Shape["N, N"], SignedInteger]
NDVector = NDArray[Shape["M"], Float] | NDArray[Shape["M"], SignedInteger]  # type: ignore

VertexId = str
Label = int
