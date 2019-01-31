import numpy as np
from model import generate_recommendations

user_address = 'u1'
already_rated = ['t0','t3','t4']
# user_address = '0x8c373ed467f3eabefd8633b52f4e1b2df00c9fe8'
# already_rated = ['0x006bea43baa3f7a6f765f14f10a1a1b08334ef45','0x5102791ca02fc3595398400bfe0e33d7b6c82267','0x68d57c9a1c35f63e2c83ee8e49a64e9d70528d25','0xc528c28fec0a90c083328bc45f587ee215760a0f']
k = 5

model_dir = './jobs/job2'

user_map = np.load(model_dir + "/user.npy")
item_map = np.load(model_dir + "/item.npy")
model = np.load(model_dir + "/model.pickle")
user_idx = np.searchsorted(user_map, user_address)
user_rated = [np.searchsorted(item_map, i) for i in already_rated]

recommendations = generate_recommendations(user_idx, user_rated, model, k, len(item_map))

tokens = [item_map[i] for i in recommendations]

print(tokens)