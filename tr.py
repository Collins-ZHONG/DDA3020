# import torch
# import jupyter

# # x = torch.rand(4,2)
# # print(x)
# # print(torch.cuda.is_available())

import numpy as np

a = np.array([8,7,11,6,5,14,10,9,11,6])
print(a.mean())
print(a.var())
print(np.linalg.norm(a - 9)**2 /10)