import json
import numpy as np


def usgs_hazard(filename, inv_t=50):
    data = json.load(open(filename))

    im_range = []
    apoes = []
    s_ranges = []
    lat_long = ""

    for i, data in enumerate(data['response']):
        # IM level
        latitude = data['metadata']['latitude']
        longitude = data['metadata']['longitude']

        lat_long = f"{latitude}-{longitude}"

        iml = data['metadata']['imt']['value']
        im_range.append(iml)

        s = data['metadata']['xvalues']
        apoe = data['data'][0]['yvalues']

        s_ranges.append(s)
        apoes.append(apoe)

    out = {
        'sat1': {}
    }

    for i, iml in enumerate(im_range):
        apoe = apoes[i]
        poe = 1 - np.exp(-np.array(apoe) * inv_t)

        out['sat1'][iml] = {
            's': s_ranges[i],
            'sites': {
                lat_long: {
                    "poe": poe,
                    "apoe": apoe,
                }
            }
        }

    return out
