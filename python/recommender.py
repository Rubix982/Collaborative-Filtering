from surprise import KNNWithMeans

# Item-based similarity
sim_options = {
    "name" : "cosine",
    "user_based" : False, # Item-based
}

algo = KNNWithMeans(sim_options=sim_options)