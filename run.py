from neural import DeepNetwork


deep_network = DeepNetwork(size=100)
deep_network.train()
deep_network.predict_from_file('data/predict.txt')

