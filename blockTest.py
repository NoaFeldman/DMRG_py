import tensornetwork as tn
from tensornetwork import BaseCharge, U1Charge, Index, BlockSparseTensor
import numpy as np
import basicOperations as bops


c_i = U1Charge([0, 1, -1]) #charges on leg i
c_j = U1Charge([0, 1, -1])   #charges on leg j
c_k = U1Charge([0, 1, -1])   #charges on leg j
#use `Index` to bundle flow and charge information
i = Index(charges=c_i, flow=False) #We use `False` and `True` to represent inflowing and outflowing arrows.
j = Index(charges=c_j, flow=False)
k = Index(charges=c_k, flow=True)
tensor = BlockSparseTensor.zeros([i, j, k], dtype=np.complex128) #creates a complex valued tensor
tensor.data = np.array(range(7))
tensor2 = BlockSparseTensor.zeros([i, j, k], dtype=np.complex128) #creates a complex valued tensor
tensor2.data[4] = 1
res = tn.block_sparse.tensordot(tensor, tensor2.conj(), ([1], [1])).data
b = 1