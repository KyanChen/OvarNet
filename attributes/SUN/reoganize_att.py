import json

import scipy.io as scio

dataFile = 'attributes.mat'
data = scio.loadmat(dataFile)
att = list(data['attributes'])
att = [x[0].item() for x in att]
json.dump(att, open('attributes_extracted.json', 'w'))
