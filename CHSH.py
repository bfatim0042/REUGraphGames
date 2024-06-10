#!/usr/bin/env python
# coding: utf-8

import numpy as np

# CHSH as a nonlocal game
prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])

dim_in_alice, dim_out_alice = 2, 2
dim_in_bob, dim_out_bob = 2, 2


def chsh_pred_mat():
    pred_mat = np.zeros((dim_in_alice, dim_out_alice, dim_in_bob, dim_out_bob))
    for a_alice in range(dim_out_alice):
        for b_bob in range(dim_out_bob):
            for x_alice in range(dim_in_alice):
                for y_bob in range(dim_in_bob):
                    if a_alice ^ b_bob == x_alice * y_bob:
                        pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
    return pred_mat


from toqito.nonlocal_games.nonlocal_game import NonlocalGame

pred_mat = chsh_pred_mat()
chsh = NonlocalGame(prob_mat, pred_mat)
# print(f"{chsh.quantum_value_lower_bound()=}")
# print(f"{chsh.nonsignaling_value()=}")
print(f"{chsh.nonsignaling_value()=}")
print(f"{chsh.quantum_value_lower_bound()=}")
print(pred_mat)
print(f"{chsh.classical_value()=}")
print(pred_mat)

# print(f"{chsh.quantum_value_lower_bound()=}")
# chsh = NonlocalGame(prob_mat, chsh_pred_mat())
print(f"{chsh.nonsignaling_value()=}")
# chsh = NonlocalGame(prob_mat, chsh_pred_mat())
print(f"{chsh.quantum_value_lower_bound()=}")

# ## Use XOR game framework
# from toqito.nonlocal_games.xor_game import XORGame

# prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
# pred_mat = np.array([[0, 0], [0, 1]])
# chsh = XORGame(prob_mat, pred_mat)

# print(f"{chsh.classical_value()=}")
# print(f"{chsh.quantum_value()=}")
# print(f"{chsh.nonsignaling_value()=}")
