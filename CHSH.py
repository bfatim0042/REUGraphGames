#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from toqito.states import basis

#The basis: {|0>, |1>}:
e_0, e_1 = basis(2, 0), basis(2, 1)


# In[38]:


epr = bell(0)


# In[45]:


#going through CHSH
prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])


# In[47]:


dim_in_alice, dim_out_alice = 2,2
dim_in_bob, dim_out_bob = 2,2


# In[49]:


pred_mat = np.zeros((dim_in_alice, dim_out_alice, dim_in_bob, dim_out_bob))


# In[52]:


for a_alice in range (dim_out_alice):
        for b_bob in range (dim_out_bob):
            for x_alice in range (dim_in_alice):
                for y_bob in range (dim_in_bob):
                    if a_alice^b_bob == x_alice * y_bob:
                        pred_mat[a_alice, b_bob, x_alice, y_bob] = 1


# In[53]:





# In[55]:


from toqito.nonlocal_games.nonlocal_game import NonlocalGame
chsh = NonlocalGame(prob_mat, pred_mat)


# In[58]:


chsh.quantum_value_lower_bound()


# In[65]:


#easier way for CHSH

from toqito.nonlocal_games.xor_game import XORGame
prob_mat = np.array([[1/4, 1/4],
                     [1/4, 1/4]])
pred_mat = np.array([[0, 0],
                    [0, 1]])
chsh = XORGame(prob_mat, pred_mat)

print(chsh.classical_value())

print(chsh.quantum_value())


# In[ ]:




