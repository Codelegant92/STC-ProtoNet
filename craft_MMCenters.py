import numpy as np
from scipy.io import savemat

L=22
D=512
mean_var=1

mean_logits=np.zeros([L,D])
mean_logits[0,0]=1.0
for k in range (1,L):
    for j in range(0, k):
    	mean_logits[k,j] = -(1.0/(L-1) + np.dot(mean_logits[k,:], mean_logits[j,:])) / mean_logits[j,j]
    	print(mean_logits[k,j])
    #print(k, np.linalg.norm(mean_logits[k,:])**2)
    mean_logits[k,k] = np.sqrt(np.abs(np.linalg.norm(mean_logits[k,:])**2))

mean_logits=mean_logits*mean_var;
mean_logits = {'mean_logits': mean_logits, 'label': 'experiment'}
savemat('meanvar1_featuredim'+str(D)+'_class'+str(L)+'.mat', mean_logits)